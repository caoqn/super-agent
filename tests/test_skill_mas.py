"""Offline adapter checks; set SKILL_MAS_CHECKOUT for pinned-runtime checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.common.skill_mas import get_adapter

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_registry_imports_without_external_benchmarks(self):
        for name in ('gaia', 'macbench', 'masbench'):
            self.assertEqual(get_adapter(name).name, name)

    def test_gaia_source_attachment_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input'
            source.mkdir()
            row = {'task_id': 'fixture', 'Question': 'Read the attachment',
                   'Final answer': 'REFERENCE_ONLY', 'file_name': 'note.txt'}
            (source / 'level_1_test_100.json').write_text(json.dumps([row]))
            files = source / 'val_files' / 'fixture'
            files.mkdir(parents=True)
            (files / 'note.txt').write_text('fixture attachment')
            workspace = Path(directory) / 'workspace'
            workspace.mkdir()
            adapter = get_adapter('gaia')
            with patch.dict(os.environ, {'GAIA_DATA_ROOT': str(source)}):
                task, = adapter.load_tasks()
                context = adapter.prepare_workspace(task, workspace)
            self.assertEqual((workspace / 'note.txt').read_text(), 'fixture attachment')
            self.assertNotIn('REFERENCE_ONLY', adapter.build_prompt(task, context))
            self.assertTrue(adapter.evaluate(task, 'FINAL ANSWER: reference only', context).success)

    def test_gaia_missing_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                get_adapter('gaia').load_tasks(Path(directory))

    def test_masbench_selection_counts_and_disjoint_ids(self):
        directory = ROOT / 'data/manifests/skill_mas/masbench'
        train = json.loads((directory / 'train_manifest.json').read_text())
        test = json.loads((directory / 'test_manifest.json').read_text())
        original = json.loads((ROOT / 'data/manifests/masbench_seed42_selection_manifest.json').read_text())
        for category in ('reasoning', 'scheduling', 'game'):
            training, testing = train[category]['ids'], test[category]['ids']
            self.assertEqual(len(set(training)), 6)
            self.assertEqual(len(set(testing)), 30)
            self.assertFalse(set(training) & set(testing))
            self.assertEqual(testing, original[category]['ids'])

    def test_assembler_rejects_wrong_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / 'upstream'
            checkout.mkdir()
            subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '--allow-empty', '-qm', 'fixture'], cwd=checkout, check=True)
            output = Path(directory) / 'runtime'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_skill_mas.py'),
                                     str(checkout), '--dataset', 'gaia', '--output-dir', str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Check out Skill-MAS commit', result.stderr)
            self.assertFalse(output.exists())


@unittest.skipUnless(os.environ.get('SKILL_MAS_CHECKOUT'), 'Set SKILL_MAS_CHECKOUT to the pinned external checkout')
class AssemblyTests(unittest.TestCase):
    def test_three_runtimes_and_native_role_bridge(self):
        for dataset in ('gaia', 'macbench', 'masbench'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'runtime'
                subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_skill_mas.py'),
                                os.environ['SKILL_MAS_CHECKOUT'], '--dataset', dataset,
                                '--output-dir', str(output)], check=True, capture_output=True)
                code = r'''
import asyncio
import importlib
from unittest.mock import patch
from Skill_MAS.bench_adapters import get_adapter
from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient
from Skill_MAS.template.sub_agent import SubAgentRequest
from methods.skill_mas import native_eval
import os

adapter = get_adapter(os.environ['TEST_DATASET'])
assert adapter.name == os.environ['TEST_DATASET']
importlib.import_module('Skill_MAS.bench_adapters.native_runtime')
assert len(list(__import__('pathlib').Path(os.environ['TEST_RUNTIME']).glob('Skill_MAS/bench_adapters/gaia.py'))) == (os.environ['TEST_DATASET'] == 'gaia')

async def check():
    request = SubAgentRequest(role_name='Alice', role_instruction='Observe private evidence',
                              user_prompt='fixture task', tool_context={'execution_mode': 'native_tool'})
    for asynchronous in (False, True):
        calls = []
        def sync_callback(payload):
            calls.append(payload)
            return ' done ', {'total_tokens': 3}
        async def async_callback(payload):
            return sync_callback(payload)
        def no_text(*args, **kwargs):
            raise AssertionError('Native execution must use the tool bridge')
        result = await request.execute_async(text_call_fn=no_text,
                    tool_call_fn=async_callback if asynchronous else sync_callback,
                    prompt_override='current subtask')
        assert result.text == 'done' and result.usage['total_tokens'] == 3
        assert len(calls) == 1
        assert calls[0]['role_name'] == 'Alice'
        assert calls[0]['task_input'] == 'current subtask'
    # Construct the actual SDK client but make no HTTP requests.
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-fixture', 'DEV_API_BASE': 'https://example.invalid/v1'}):
        client = AsyncOpenAIClient(model='qwen3.5-plus')
        assert client.base_url == 'https://example.invalid/v1'
        await client.aclose()
asyncio.run(check())
'''
                env = {**os.environ, 'SUPER_AGENT_ROOT': str(ROOT), 'TEST_DATASET': dataset,
                       'TEST_RUNTIME': str(output), 'PYTHONPATH': os.pathsep.join((str(output), str(ROOT)))}
                subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    unittest.main()
