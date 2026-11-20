import asyncio
from types import SimpleNamespace

import pytest

from methods.Evoteam.bootstrap import configure_source
configure_source()
from benchmarks.adapter import EnvContext
from methods.Evoteam.macbench.adapter import MacBenchAdapter, _TaskActionBudget
from core import llm
from core.api_resilience import APIReliabilityTracker
from core.runner import Runner
from core.task_budget import TaskBudget, TaskBudgetExceeded, prompt_token_reservation


def budget(**changes):
    limits = dict(max_tokens=100000, max_model_calls=100, max_tool_calls=30,
                  max_env_actions=20, max_seconds=7200)
    limits.update(changes)
    return TaskBudget(limits)


def response(prompt=100, output=20, text='FINAL OUTPUT: Saved the verified partial changes.'):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=output),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
    )


def test_concurrent_reservations_keep_emergency_allowance():
    b = budget(max_tokens=1000)
    first, _ = b.admit_model(400, 200)
    second, output = b.admit_model(100, 200)
    assert output == 50
    with pytest.raises(TaskBudgetExceeded, match='tokens'):
        b.admit_model(1, 1)
    assert b.reserved_tokens == 750
    b.settle_model(first, response(400, 200))
    b.settle_model(second, response(100, 50))
    final, output = b.admit_model(100, 150, emergency=True)
    assert output == 150
    b.settle_model(final, response(100, 150))
    assert b.snapshot()['total_tokens'] == 1000
    assert b.reserved_tokens == 0


def test_model_call_limit_keeps_one_final_call():
    b = budget(max_model_calls=3)
    for _ in range(2):
        reserved, _ = b.admit_model(100, 20)
        b.settle_model(reserved, response())
    with pytest.raises(TaskBudgetExceeded, match='model_calls'):
        b.admit_model(100, 20)
    reserved, _ = b.admit_model(100, 20, emergency=True)
    b.settle_model(reserved, response())
    with pytest.raises(TaskBudgetExceeded, match='model_calls'):
        b.admit_model(100, 20, emergency=True)
    assert b.model_call_attempts == 3


def test_tool_limit_uses_minimum_of_tool_and_environment_limits():
    b = budget(max_tool_calls=5, max_env_actions=2)
    b.admit()
    b.admit()
    with pytest.raises(TaskBudgetExceeded, match='tool_calls'):
        b.admit()
    assert b.calls == 2


def test_time_reserve_stops_work_but_allows_final_model_call():
    b = budget(max_seconds=100)
    b.started -= 91
    assert b.submission_reason() == 'wall_seconds'
    with pytest.raises(TaskBudgetExceeded):
        b.admit()
    reserved, _ = b.admit_model(100, 20, emergency=True)
    b.settle_model(reserved, response())
    b.started -= 10
    with pytest.raises(TaskBudgetExceeded, match='wall_seconds'):
        b.admit_model(100, 20, emergency=True)


def test_llm_retry_counts_attempts_and_unknown_usage(monkeypatch):
    b = budget(max_model_calls=3)
    tracker = APIReliabilityTracker()
    tracker.macbench_budget = b
    attempts = []

    async def fake_completion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError('bad gateway')
        return response()

    monkeypatch.setattr(llm.litellm, 'acompletion', fake_completion)
    monkeypatch.setattr(llm, '_circuit', lambda: None)
    monkeypatch.setattr(llm, 'BASE_DELAY', 0)
    monkeypatch.setattr(llm, 'JITTER_RANGE', 0)
    asyncio.run(llm.complete('test', [{'role': 'user', 'content': 'task'}],
                             max_tokens=20, api_tracker=tracker))
    assert b.model_call_attempts == 2
    assert b.model_calls == 1
    assert b.uncertain_tokens > 0
    assert b.input_tokens == 100
    assert b.output_tokens == 20
    assert b.reserved_tokens == 0


def test_cancelled_model_call_keeps_conservative_usage(monkeypatch):
    b = budget()
    tracker = APIReliabilityTracker()
    tracker.macbench_budget = b

    async def cancelled(**kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(llm.litellm, 'acompletion', cancelled)
    monkeypatch.setattr(llm, '_circuit', lambda: None)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(llm.complete('test', [{'role': 'user', 'content': 'task'}],
                                 max_tokens=20, api_tracker=tracker))
    assert b.reserved_tokens == 0
    assert b.uncertain_tokens > 0


def test_runner_budget_emergency_uses_chairman_and_bounded_prompt(tmp_path, monkeypatch):
    runner = Runner(pool_dir=tmp_path, chairman_name='chairman',
                    output_contract=MacBenchAdapter().output_contract())
    runner.task_budget = budget()
    runner.api_tracker.macbench_budget = runner.task_budget
    runner._task = 'Preserve completed changes.'
    runner._forced_termination = 'macbench_budget_exceeded:tokens'
    runner.agents['chairman'] = SimpleNamespace(
        config=SimpleNamespace(name='chairman', model='test', temperature=0, max_tokens=16384),
        messages=[{'role': 'user', 'content': 'task'},
                  {'role': 'assistant', 'content': 'Earlier evidence. ' * 100000},
                  {'role': 'tool', 'content': 'Saved record A; record B remains incomplete.'}],
        build_system_prompt=lambda: 'Long prompt ' * 10000,
    )
    captured = {}

    async def fake_complete(**kwargs):
        captured.update(kwargs)
        return response()

    monkeypatch.setattr(llm, 'complete', fake_complete)
    asyncio.run(runner._emergency_finalize_after_forced_termination())
    assert captured['budget_emergency'] is True
    assert 'tools' not in captured
    assert captured['max_tokens'] == 2048
    assert prompt_token_reservation(captured['messages']) + captured['max_tokens'] <= runner.task_budget.reserve_tokens
    assert 'record B remains incomplete' in captured['messages'][1]['content']
    assert runner._finalization_mode == 'chairman_budget_emergency'
    assert runner._final_output_locked
    assert runner._result == 'Saved the verified partial changes.'


def test_native_budget_callback_wakes_runner(tmp_path):
    async def scenario():
        b = _TaskActionBudget(budget(max_model_calls=2).limits)
        runner = Runner(pool_dir=tmp_path, chairman_name='chairman', max_seconds=7200)
        runner.task_budget = b
        MacBenchAdapter().configure_native_runner({}, EnvContext({'task_budget': b}), runner)
        b.model_call_attempts = 1
        with pytest.raises(TaskBudgetExceeded):
            b.admit_model(100, 20)
        await asyncio.sleep(0)
        assert runner._done.is_set()
        assert runner._forced_termination == 'macbench_budget_exceeded:model_calls'
    asyncio.run(scenario())


def test_full_runner_stops_and_submits_when_model_budget_reached(tmp_path, monkeypatch):
    import yaml
    agent_dir = tmp_path / 'chairman'
    agent_dir.mkdir()
    (agent_dir / 'config.yaml').write_text(yaml.safe_dump(dict(
        name='chairman', model='test', max_steps=50, max_tokens=100,
        tools=[], role='chairman',
    )))
    (agent_dir / 'prompt.md').write_text('Complete the task.')
    b = _TaskActionBudget(budget(max_model_calls=2).limits)
    runner = Runner(pool_dir=tmp_path, chairman_name='chairman',
                    output_contract=MacBenchAdapter().output_contract(), max_seconds=7200)
    runner.task_budget = b
    runner.api_tracker.macbench_budget = b
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # Real llm.complete still performs admission/settlement.
            return response(text='Task evidence gathered.')
        return response(text='FINAL OUTPUT: Current work submitted.')

    monkeypatch.setattr(llm.litellm, 'acompletion', fake_completion)
    monkeypatch.setattr(llm, '_circuit', lambda: None)
    monkeypatch.setattr(runner, '_CHAIRMAN_EXIT_GRACE_SECONDS', 0)

    async def scenario():
        MacBenchAdapter().configure_native_runner({}, EnvContext({'task_budget': b}), runner)
        return await asyncio.wait_for(runner.run('Complete the task.'), timeout=5)

    result = asyncio.run(scenario())
    assert result.output == 'Current work submitted.'
    assert result.metadata['finalization_mode'] == 'chairman_budget_emergency'
    assert len(calls) == 2
    assert b.model_call_attempts == 2
    assert b.reserved_tokens == 0
