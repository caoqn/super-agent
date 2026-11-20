#!/usr/bin/env python3
"""Run fixed train/test splits through an externally prepared Skill_MAS package."""
import argparse,hashlib,json,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--upstream',type=Path,required=True);p.add_argument('--benchmark',choices=['gaia','masbench','macbench'],required=True)
 p.add_argument('--phase',choices=['train','test'],required=True);p.add_argument('--data-root',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--skill',type=Path)
 p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 upstream=a.upstream.resolve();config=json.loads((REPO/f'methods/skillmas/config/{a.benchmark}.json').read_text())
 if not (upstream/'REPRODUCTION.json').is_file():p.error('Run prepare_skillmas.py first')
 os.environ['SKILL_MAS_RESULTS_ROOT']=str(a.output.resolve())
 os.environ['SKILL_MAS_TRUST_ENV']='0'
 os.environ.pop('SKILL_MAS_MAX_TOKENS',None)
 envkey={'gaia':'GAIA_DATA_ROOT','masbench':'MASBENCH_ROOT','macbench':'MAC_BENCH_ROOT'}[a.benchmark]
 os.environ[envkey]=str(a.data_root.resolve());os.environ['MASBENCH_SPLIT_DIR']=str(REPO/'methods/skillmas/splits/masbench')
 sys.path.insert(0,str(upstream.parent))
 from Skill_MAS.bench_adapters import get_adapter
 tasks=get_adapter(a.benchmark).load_tasks(None,split=a.phase);ids=[str(t.task_id) for t in tasks]
 if len(ids)!=config[f'{a.phase}_tasks'] or len(set(ids))!=len(ids):raise ValueError('Unexpected split size or duplicate task IDs')
 if a.benchmark=='gaia':
  expected=json.loads((REPO/f'methods/skillmas/splits/gaia/{a.phase}.json').read_text())
  if ids!=expected:raise ValueError('GAIA task IDs/order differ from frozen split')
 if a.benchmark=='masbench':
  manifest=json.loads((REPO/f'methods/skillmas/splits/masbench/{a.phase}_manifest.json').read_text())
  expected=[f'{category}__{task_id}' for category in ('reasoning','scheduling','game') for task_id in manifest[category]['ids']]
  if ids!=expected:raise ValueError('MASBench task IDs/order differ from frozen manifest')
 if a.benchmark=='macbench':
  ev=next(e for e in json.loads((REPO/'methods/skillmas/experiment_evidence.json').read_text()) if e['benchmark']=='macbench')
  if a.phase=='test':
   if ids!=ev['test_manifest']['task_ids']:raise ValueError('MACBench task IDs/order mismatch')
   if any(t.raw['world_revision']!=ev['test_manifest']['task_revisions'][t.task_id] for t in tasks):raise ValueError('MACBench data revision mismatch')
  elif a.phase=='train' and len(ids)!=config['train_tasks']:raise ValueError('MACBench dev task count mismatch')
 if a.phase=='test' and not a.validate_only and (not a.skill or not a.skill.is_file()):p.error('test requires --skill pointing to a frozen SKILL.md')
 print(json.dumps({'benchmark':a.benchmark,'phase':a.phase,'tasks':len(ids),'configuration':config},indent=2))
 if a.validate_only:return
 if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('OPENAI_BASE_URL'):p.error('Set OPENAI_API_KEY and OPENAI_BASE_URL')
 from Skill_MAS.utils.config import runs_dir,skills_evolution_dir
 from Skill_MAS.skill_mas import openai_async_client
 # A run-local model file avoids concurrent runs overwriting shared configuration.
 model_config=a.output.resolve()/'.skillmas-config'/a.benchmark/(a.run_id+'.json')
 model_config.parent.mkdir(parents=True,exist_ok=True)
 model_config.write_text(json.dumps({config['model']:{'api_key':'${OPENAI_API_KEY}','temperature':config['temperature'],'max_tokens':config['max_output_tokens'],'input_per_1m':config['input_per_1m'],'output_per_1m':config['output_per_1m']}},indent=2))
 openai_async_client.MODEL_CONFIG_PATH=model_config
 from Skill_MAS.evolution import native_eval
 native_eval._NATIVE_TASK_TIMEOUT_S[a.benchmark]=config['base_timeout_seconds']
 native_eval._TIMEOUT_WAIT_WINDOW_S=config['additional_wall_window_seconds']
 bench_id=f'skillmas_{a.benchmark}'
 runroot=runs_dir(a.benchmark,config['model'])/bench_id/a.run_id;runroot.mkdir(parents=True,exist_ok=True)
 effective={'configuration':config,'phase':a.phase,'task_ids':ids,'skill_sha256':hashlib.sha256(a.skill.read_bytes()).hexdigest() if a.skill else None,'prepared_source':json.loads((upstream/'REPRODUCTION.json').read_text())}
 record=runroot/'effective_config.json'
 if record.exists() and json.loads(record.read_text())!=effective:raise ValueError('Resume configuration changed; use a new run ID')
 record.write_text(json.dumps(effective,indent=2))
 if a.phase=='train':
  from Skill_MAS.core.pipeline import evolve
  evolve(domain=a.benchmark,bench_backend=a.benchmark,bench_id=bench_id,run_id=a.run_id,rounds=1,k_trajectories=config['k_trajectories'],native_split='train',max_problems=0,max_steps=config['max_steps'],max_concurrency=config['train_concurrency'],agent_llm=config['model'],optimizer_llm=config['model'],user_llm=config['model'],evaluator_llm=config['model'])
 else:
  from Skill_MAS.evolution.native_eval import run_native_evaluation_round
  frozen=runroot/'frozen_skill/SKILL.md';frozen.parent.mkdir(exist_ok=True)
  if frozen.exists() and frozen.read_bytes()!=a.skill.read_bytes():raise ValueError('Frozen skill changed')
  frozen.write_bytes(a.skill.read_bytes())
  run_native_evaluation_round(bench_backend=a.benchmark,bench_id=bench_id,run_id=a.run_id,round_idx=0,task_ids=ids,agent_llm=config['model'],runs_dir=runs_dir(a.benchmark,config['model']),skills_evolution_dir=skills_evolution_dir(a.benchmark,config['model']),max_concurrency=config['test_concurrency'],skill_file=frozen,split='test')
if __name__=='__main__':main()
