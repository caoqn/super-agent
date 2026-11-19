import asyncio
import json
from argparse import Namespace
from pathlib import Path

from benchmarks.adapter_macbench import (
    MacBenchAdapter,
    _load_catalog,
    _set_runtime_pool_tools,
)
from core.tool_registry import ToolRegistry
from tools import macbench


import os
import pytest
MAC_BENCH_ROOT = Path(os.environ.get("MAC_BENCH_ROOT", "external/MAC_Bench-main"))
pytestmark = pytest.mark.skipif(not MAC_BENCH_ROOT.exists(), reason="Set MAC_BENCH_ROOT for integration tests")


def test_load_public_macbench_test_and_dev_splits():
    adapter = MacBenchAdapter()
    common = {"macbench_root": MAC_BENCH_ROOT, "category": None, "max_items": 0}

    test_rows = adapter.load_dataset(Namespace(split="test", **common))
    dev_rows = adapter.load_dataset(Namespace(split="dev", **common))

    assert len(test_rows) == 48
    assert len(dev_rows) == 6
    assert {row["_macbench_split"] for row in test_rows} == {"test"}
    assert all(row["_macbench_instruction"].strip() for row in test_rows + dev_rows)
    assert all("_macbench_bundle" in row for row in test_rows + dev_rows)
    assert all(row["_macbench_budget"]["max_tokens"] > 0 for row in test_rows + dev_rows)
    assert all("_macbench_budget_source" in row for row in test_rows + dev_rows)


def test_loaded_task_does_not_put_bundle_path_in_agent_prompt():
    adapter = MacBenchAdapter()
    item = adapter.load_dataset(Namespace(
        split="test", macbench_root=MAC_BENCH_ROOT,
        category=None, max_items=1,
    ))[0]
    prompt = adapter.build_task(item, None, [])

    assert item["_macbench_bundle"] not in prompt
    assert item["_macbench_revision"] not in prompt
    assert item["_macbench_instruction"] in prompt
    assert "macbench(action='describe'" in prompt


def test_catalog_loader_checks_public_revision_and_bundle():
    data_dir, rows, public = _load_catalog(MAC_BENCH_ROOT)

    assert data_dir == (MAC_BENCH_ROOT / "data").resolve()
    assert len(rows) == len(public) == 54


def test_output_contract_uses_stateful_environment_completion():
    contract = MacBenchAdapter().output_contract()

    assert contract.name == "macbench"
    assert contract.submission_type == "stateful_environment_completion"
    assert not contract.answer_agent_enabled


def test_meta_team_tool_registry_loads_macbench_bridge():
    tool = ToolRegistry(str(Path(__file__).resolve().parents[1] / "methods/Evoteam/runtime/tools")).get("macbench")

    assert tool is not None
    assert tool.to_openai_tool()["function"]["name"] == "macbench"


def test_runtime_team_tool_scope_only_changes_disposable_configs(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "pool.yaml").write_text(
        "chairman: planner\nsettings:\n  answer_agent: answer_agent\n",
        encoding="utf-8",
    )
    config = runtime / "implementer" / "config.yaml"
    answer_config = runtime / "answer_agent" / "config.yaml"
    config.parent.mkdir(parents=True)
    answer_config.parent.mkdir(parents=True)
    config.write_text("name: implementer\ntools: [bash, web_search]\n", encoding="utf-8")
    answer_config.write_text("name: answer_agent\ntools: [bash]\n", encoding="utf-8")

    _set_runtime_pool_tools(config.parents[1])

    import yaml

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["tools"] == ["macbench"]
    assert yaml.safe_load(answer_config.read_text(encoding="utf-8"))["tools"] == []


def test_macbench_tool_bridge_lists_describes_and_dispatches_task_tools():
    class Tool:
        name = "update_record"
        description = "Update one public record."
        input_schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    class Environment:
        def call(self, name, arguments):
            return {"tool": name, "arguments": arguments}

    class Handle:
        view = type("View", (), {"tools": (Tool(),)})()
        environment = Environment()

    token = macbench.set_active_task(Handle())
    try:
        listed = json.loads(asyncio.run(macbench.execute(action="list_tools")))
        described = json.loads(asyncio.run(macbench.execute(
            action="describe", tool_name="update_record",
        )))
        result = json.loads(asyncio.run(macbench.execute(
            action="call", tool_name="update_record", arguments={"id": "r1"},
        )))
    finally:
        macbench.clear_active_task(token)

    assert listed[0]["name"] == "update_record"
    assert described["parameters"] == Tool.input_schema
    assert result == {"tool": "update_record", "arguments": {"id": "r1"}}
