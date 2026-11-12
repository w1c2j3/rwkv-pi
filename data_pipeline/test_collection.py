"""Optional native-Pi integration against a local synthetic SSE endpoint; zero paid calls.

Set RWKV_PIPELINE_TEST_RUNTIME to the existing private sandbox collector runtime,
and RWKV_PI_SOURCE to its pinned native Pi source. No real API credentials are read.
"""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .test_products import ProductTests
from .products import write, read, plan_trajectories, profile_identity
from .execution import collect_plan


@unittest.skipUnless(os.environ.get('RWKV_PIPELINE_TEST_RUNTIME'), 'Private native runtime not configured')
class CollectionTest(unittest.TestCase):
    def test_two_models_same_snapshot_followups_and_shared_budget(self):
        fixture = ProductTests('test_question_export_preserves_bytes_and_never_runs_checks')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                is_tool_result = body['messages'][-1]['role'] == 'tool'
                content = {'role': 'assistant', 'content': '这是本地接口测试结果，不是真实训练轨迹。'} if is_tool_result else {
                    'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'fixture-call-' + str(len(requests)),
                                                      'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"fixture.txt"}'}}]}
                base = {'id': 'fixture-response', 'object': 'chat.completion.chunk', 'created': 1, 'model': body['model']}
                for event in [{**base, 'choices': [{'index': 0, 'delta': content, 'finish_reason': None}]},
                              {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop' if is_tool_result else 'tool_calls'}]},
                              {**base, 'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 40}}]:
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = fixture.root
        runtime = Path(os.environ['RWKV_PIPELINE_TEST_RUNTIME'])
        write(root / 'key.json', 'engineering fixture only')
        profiles, paths = [], []
        endpoint = f'http://127.0.0.1:{server.server_port}'
        pricing = {model: {'provider': 'engineering', 'baseUrl': endpoint, 'input': 0.1, 'output': 0.1}
                   for model in ['fixture-a', 'fixture-b']}
        for model in pricing:
            profile = {**fixture.profile, 'model': model, 'baseUrl': endpoint, 'thinking': 'off',
                       'contextWindow': 32768, 'maxTokens': 512,
                       'credentialFile': str(root / 'key.json'),
                       'budget': {'cnyLimit': 1, 'incrementCny': 1, 'ledgerPath': str(root / 'budget.json'),
                                  'ownerLock': str(root / 'budget.lock'), 'pricing': pricing},
                       'execution': {'node': '/home/chase/.local/node/bin/node',
                                     'launcher': str(Path(__file__).resolve().parents[1] / 'scripts/run-pi-source.mjs'),
                                     'runtime': str(runtime), 'piSource': os.environ['RWKV_PI_SOURCE'],
                                     'budgetModule': str(runtime / 'pipeline/paid-trial-budget.mjs'),
                                     'harnessModule': str(runtime / 'pipeline/teacher-followup-harness.mjs'),
                                     'workflowConcurrency': 1, 'apiConcurrency': 1, 'maxRequests': 8, 'timeoutMs': 60000}}
            p = root / (model + '.json')
            write(p, profile)
            profiles.append(profile)
            paths.append(p)
        plan_trajectories(fixture.questions, profiles, root / 'collection')
        result = collect_plan(root / 'collection', paths)
        details = [read(p) for p in (root / 'collection/teachers').glob('*/experiment.json')]
        self.assertEqual(result['state'], 'collected_awaiting_review', details)
        self.assertEqual(result['trainingRows'], 0)
        self.assertEqual(len(requests), 8)
        for profile in profiles:
            key, _ = profile_identity(profile)
            experiment = read(root / 'collection/teachers' / key[:16] / 'experiment.json')
            self.assertEqual(experiment['results'][0]['status'], 'passed')
            self.assertEqual(experiment['results'][0]['rawToolCalls'], 2)
            run = read(experiment['results'][0]['run'])
            self.assertEqual(run['teacherProfileId'], key)
            self.assertEqual(run['model']['id'], profile['model'])
            self.assertTrue(run['conversation']['allTurnsCompleted'])
            self.assertFalse(run['trainingEligible'])
        ledger = read(root / 'budget.json')
        self.assertEqual(len(ledger['requests']), 8)
        self.assertEqual(ledger['budgetMicroCny'], 1000000)
        self.assertEqual(set(r['arm'] for r in ledger['requests']), set(pricing))
        self.assertFalse(ledger['halted'])


if __name__ == '__main__':
    unittest.main()
