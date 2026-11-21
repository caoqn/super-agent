"""Offline integration checks for concurrent delegation and shared accounting."""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from types import SimpleNamespace
import json
import time
import unittest

from mac_bench.evaluation.episodes.budget import BudgetExceeded, EpisodeBudget
from mac_bench.evaluation.episodes.identity import call_identity
from mac_bench.evaluation.episodes.recording import RecordingClient
from mac_bench.runtime.agents.workflow import AgentBudget
from mac_bench.runtime.llm.client import ChatResponse
from methods.manager3.macbench.agent import PromptManager3, WORKERS


def response(content='', name=None, args=None):
    calls = () if name is None else ({'id': 'call-test', 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(args)}},)
    return ChatResponse(content, 'offline', {'prompt_tokens': 10, 'completion_tokens': 5},
                        finish_reason='tool_calls' if calls else 'stop', tool_calls=calls)


class Tool:
    name = 'inspect'

    def as_function(self):
        return {'type': 'function', 'function': {'name': self.name,
            'parameters': {'type': 'object', 'properties': {}}}}


class Environment:
    def __init__(self):
        self.identities = []
        self.active = self.peak = 0

    def call(self, name, args):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.identities.append(call_identity())
        time.sleep(.01)
        self.active -= 1
        return {'verified': True}


class ScriptedClient:
    def __init__(self, fail=False):
        self.calls = {}
        self.barrier = Barrier(3, timeout=5)
        self.lock = Lock()
        self.fail = fail
        self.worker1_recalled = False

    def complete(self, messages, **kwargs):
        agent = call_identity()['agent_id']
        with self.lock:
            count = self.calls[agent] = self.calls.get(agent, 0) + 1
        if agent == 'manager':
            if count == 1:
                assert [tool['function']['name'] for tool in kwargs['tools']] == ['delegate_tasks']
                return response(name='delegate_tasks', args={'assignments': [
                    {'agent_id': worker, 'task': 'Inspect your assigned component'} for worker in WORKERS]})
            if count == 2:
                reports = json.loads(messages[-1].content)['reports']
                assert len(reports) == 3 and all(row['summary'] for row in reports)
                return response(name='delegate_tasks', args={'assignments': [
                    {'agent_id': 'worker-1', 'task': 'Follow up using your earlier observations'}]})
            if count == 3:
                return response(name='inspect', args={})
            return response('Task verified')
        if count == 1:
            self.barrier.wait()
            if self.fail:
                if agent == 'worker-1':
                    raise RuntimeError('Injected worker failure')
                time.sleep(.05)
        if count == 3:
            assert agent == 'worker-1'
            assert any('round 1' in str(m.content) for m in messages)
            assert any(m.role == 'tool' for m in messages)
            self.worker1_recalled = True
        return response(name='inspect', args={}) if count % 2 else response('Verified component')


class TeamIntegrationTests(unittest.TestCase):
    def context(self, directory, client, max_calls=100):
        budget = EpisodeBudget(AgentBudget(max_model_calls=max_calls, max_tool_calls=100,
            max_env_actions=100, max_tokens=10000, max_wall_seconds=60))
        environment = Environment()
        events = []
        context = SimpleNamespace(task=SimpleNamespace(instruction='Fix and verify the example',
            interaction_instructions='Use inspect.', tools=[Tool()]),
            environment=environment, budget=budget, event=events.append,
            client=RecordingClient(client, Path(directory) / 'calls', budget=budget))
        return context, events

    def test_parallel_workers_persistent_context_and_aggregate_accounting(self):
        with TemporaryDirectory() as directory:
            client = ScriptedClient()
            context, events = self.context(directory, client)
            outcome = PromptManager3().run(context)
            self.assertEqual(outcome['stop_reason'], 'finished')
            self.assertTrue(client.worker1_recalled)
            self.assertEqual(client.calls, {'manager': 4, 'worker-1': 4, 'worker-2': 2, 'worker-3': 2})
            self.assertEqual(context.environment.peak, 1)
            self.assertEqual({row['agent_id'] for row in context.environment.identities},
                             {'manager', *WORKERS})
            self.assertEqual(context.budget.snapshot()['input_tokens'], 120)
            self.assertEqual(context.budget.snapshot()['output_tokens'], 60)
            records = [json.loads(p.read_text()) for p in (Path(directory) / 'calls').glob('*.json')]
            self.assertEqual(len(records), 12)
            self.assertEqual(sum(row['usage']['prompt_tokens'] for row in records), 120)
            self.assertEqual({row['identity']['agent_id'] for row in records}, {'manager', *WORKERS})
            active = set()
            peak = 0
            for event in events:
                if event['kind'] == 'agent_start':
                    active.add(event['agent_id'])
                    peak = max(peak, len(active))
                elif event['kind'] == 'agent_end':
                    active.remove(event['agent_id'])
            self.assertEqual(peak, 4)
            self.assertFalse(active)

    def test_worker_failure_joins_and_prevents_late_environment_writes(self):
        with TemporaryDirectory() as directory:
            context, events = self.context(directory, ScriptedClient(fail=True))
            with self.assertRaisesRegex(RuntimeError, 'Injected worker failure'):
                PromptManager3().run(context)
            self.assertFalse(context.environment.identities)
            self.assertEqual(sum(e['kind'] == 'agent_start' for e in events),
                             sum(e['kind'] == 'agent_end' for e in events))
            self.assertEqual(context.budget._model_inflight, 0)

    def test_one_shared_budget_stops_the_whole_team(self):
        with TemporaryDirectory() as directory:
            context, events = self.context(directory, ScriptedClient(), max_calls=1)
            with self.assertRaisesRegex(BudgetExceeded, 'model_calls'):
                PromptManager3().run(context)
            self.assertEqual(context.budget.snapshot()['model_call_attempts'], 1)
            self.assertFalse(context.environment.identities)
            self.assertEqual(context.budget._model_inflight, 0)
            self.assertEqual(sum(e['kind'] == 'agent_start' for e in events),
                             sum(e['kind'] == 'agent_end' for e in events))

    def test_invalid_and_premature_tools_get_feedback_before_delegation(self):
        class ValidationClient(ScriptedClient):
            def complete(self, messages, **kwargs):
                if call_identity()['agent_id'] == 'manager' and not self.calls.get('manager'):
                    previous = [m for m in messages if m.role == 'tool']
                    if not previous:
                        return response(name='inspect', args={})
                    if len(previous) == 1:
                        assert 'Unknown tool' in previous[-1].content
                        return response(name='delegate_tasks', args={'assignments': [
                            {'agent_id': 'worker-1', 'task': 'Inspect'}]})
                    if len(previous) == 2:
                        assert 'all three' in previous[-1].content
                return super().complete(messages, **kwargs)

        with TemporaryDirectory() as directory:
            context, events = self.context(directory, ValidationClient())
            self.assertEqual(PromptManager3().run(context)['stop_reason'], 'finished')
            self.assertEqual(len(context.environment.identities), 5)


if __name__ == '__main__':
    unittest.main()
