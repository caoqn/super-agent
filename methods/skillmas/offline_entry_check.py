"""Offline entry smoke: real adapters and client configuration, stub paid execution."""
import hashlib, json, os, runpy, sys, tempfile
from pathlib import Path
from unittest.mock import patch

repo=Path(__file__).resolve().parents[2]
upstream=Path(os.environ['SKILLMAS_TEST_UPSTREAM']).resolve()
benchmark=os.environ['SKILLMAS_TEST_BENCHMARK']
phase=os.environ['SKILLMAS_TEST_PHASE']
data_root=os.environ['SKILLMAS_TEST_DATA_ROOT']
sys.path.insert(0,str(upstream.parent))
from Skill_MAS.core import pipeline
from Skill_MAS.evolution import native_eval
from Skill_MAS.skill_mas import openai_async_client
config=json.loads((repo/'methods/skillmas/config'/f'{benchmark}.json').read_text())
skill=repo/'methods/skillmas/artifacts'/benchmark/'SKILL.md'
evidence=next(x for x in json.loads((repo/'methods/skillmas/experiment_evidence.json').read_text()) if x['benchmark']==benchmark)
assert hashlib.sha256(skill.read_bytes()).hexdigest()==evidence['skill_sha256']
called=[]
def fake_train(**kwargs):
 assert kwargs['domain']==benchmark and kwargs['bench_backend']==benchmark
 assert kwargs['k_trajectories']==config['k_trajectories'] and kwargs['rounds']==1
 called.append(kwargs)
def fake_test(**kwargs):
 assert len(kwargs['task_ids'])==config['test_tasks']
 assert kwargs['skill_file'].read_bytes()==skill.read_bytes()
 called.append(kwargs)
from Skill_MAS.utils import config as paths_config
with tempfile.TemporaryDirectory() as temp, patch.object(pipeline,'evolve',fake_train), patch.object(native_eval,'run_native_evaluation_round',fake_test), patch.object(paths_config,'RESULTS_ROOT',Path(temp)):
 os.environ['OPENAI_API_KEY']='offline-dummy-key';os.environ['OPENAI_BASE_URL']='https://example.invalid/v1'
 sys.argv=['run_skillmas.py','--upstream',str(upstream),'--benchmark',benchmark,'--phase',phase,'--data-root',data_root,'--output',temp,'--run-id','offline']
 if phase=='test':sys.argv+=['--skill',str(skill)]
 runpy.run_path(str(repo/'scripts/run_skillmas.py'),run_name='__main__')
 assert len(called)==1
 row=openai_async_client._load_model_config()[config['model']]
 assert row['temperature']==config['temperature'] and row['max_tokens']==config['max_output_tokens']
 assert row['input_per_1m']==config['input_per_1m'] and row['output_per_1m']==config['output_per_1m']
 assert native_eval._task_wall_timeout(benchmark)==config['base_timeout_seconds']+config['additional_wall_window_seconds']
 print('OFFLINE ENTRY OK',benchmark,phase)
