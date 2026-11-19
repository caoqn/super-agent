import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'gaia_adapter', Path(__file__).resolve().parents[1] / 'benchmarks/gaia/adapters/super_agent/adapter.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class Client:
    def __init__(self, ticket=None):
        self.ticket = ticket
        self.stops = []

    def call(self, method, path, body=None):
        if method == 'POST' and path == 'benchmark/tasks':
            assert 'SECRET_REFERENCE' not in body['prompt']
            return {'trigger_id': 'benchmark:r/q', 'thread': 'benchmark:r/q', 'result': 'stored'}
        if path.startswith('threads/'):
            return {'tickets': [{'id': 't'}] if self.ticket else [], 'timeline': []}
        if path == 'tickets/t/stop':
            self.stops.append('t')
            self.ticket = dict(self.ticket, state='ended')
            return {'stopped': True}
        if path == 'tickets/t':
            return self.ticket
        raise AssertionError(path)


class AdapterTests(unittest.TestCase):
    def run_item(self, client, timeout=1):
        with tempfile.TemporaryDirectory() as directory:
            return adapter.run_item(client, {'task_id': 'q', '_level': 1,
                'Question': 'What is the answer?', 'Final answer': 'SECRET_REFERENCE'},
                'r', Path(directory), Path(directory), timeout, .001)

    def test_answer_is_one_line_and_final_has_priority(self):
        self.assertEqual(adapter.extract_final_answer('Answer: 12\nMethod: details'), '12')
        self.assertEqual(adapter.extract_final_answer('FINAL ANSWER: 42\r\nAnswer: 12'), '42')
        self.assertEqual(adapter.extract_final_answer('FINAL ANSWER: \nMethod: details'), '')
        self.assertEqual(adapter.extract_final_answer('GAIA Level 1 answer: **0.1777 m³**\nSource: paper'), '0.1777 m³')
        self.assertEqual(adapter.extract_final_answer('**Final answer: Indira Gandhi** (April 1977)'), 'Indira Gandhi')
        self.assertEqual(adapter.extract_final_answer('GAIA L3 (task x): answer = **86**'), '86')

    def test_ready_review_is_not_completed(self):
        client = Client({'state': 'open', 'stage': 'person', 'action': 'review',
            'assignments': [{'report': {'state': 'ready', 'summary': 'Answer: 42'}}]})
        result = self.run_item(client)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(client.stops, ['t'])
        self.assertIsNone(result['human_interventions'])

    def test_completed_ticket_is_not_stopped(self):
        client = Client({'state': 'ended', 'stage': 'ended', 'ended': {'outcome': 'completed'},
            'assignments': [{'report': {'state': 'ready', 'summary': 'Answer: 42'}}]})
        result = self.run_item(client)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['prediction'], '42')
        self.assertEqual(client.stops, [])

    def test_unresolved_routing_fails_cleanup(self):
        result = self.run_item(Client(), timeout=.002)
        self.assertEqual(result['status'], 'timeout')
        self.assertTrue(result['cleanup_failed'])

    def test_delivery_failure_not_worker_progress(self):
        result = self.run_item(Client({'state': 'open', 'stage': 'person', 'action': 'delivery',
            'assignments': [{'report': {'state': 'progress'}}]}))
        self.assertEqual(result['status'], 'delivery_failed')


if __name__ == '__main__':
    unittest.main()
