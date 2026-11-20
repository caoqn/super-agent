#!/usr/bin/env python3
"""Prepare a pinned external Skill_MAS package; never overwrite a checkout."""
import argparse,hashlib,json,shutil,subprocess,tempfile,zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--archive',type=Path);a=p.parse_args()
 dst=a.destination.resolve()
 if dst.name!='Skill_MAS':p.error('destination basename must be Skill_MAS (Python package name)')
 if dst.exists():p.error('destination already exists; choose a fresh directory')
 lock=json.loads((REPO/'methods/skillmas/upstream.json').read_text());dst.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
  stage=Path(tmp)/'Skill_MAS'
  if a.archive:
   if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=lock['archive_sha256']:raise ValueError('Archive checksum mismatch')
   with zipfile.ZipFile(a.archive) as z:
    if z.comment.decode()!=lock['commit']:raise ValueError('Archive revision mismatch')
    for n in z.namelist():
     if n.endswith('/'):continue
     rel=Path(n).parts[1:];q=stage.joinpath(*rel)
     if stage.resolve() not in q.resolve().parents:raise ValueError('Unsafe archive member')
     q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(n))
  else:
   subprocess.run(['git','clone',lock['repository'],str(stage)],check=True)
   subprocess.run(['git','checkout','--detach',lock['commit']],cwd=stage,check=True)
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=stage,text=True).strip()==lock['commit']
  patch=REPO/'methods/skillmas/patches/runtime.patch'
  subprocess.run(['git','apply','--check',str(patch)],cwd=stage,check=True)
  subprocess.run(['git','apply',str(patch)],cwd=stage,check=True)
  adapters=stage/'bench_adapters';adapters.mkdir()
  common=REPO/'benchmarks/common/adapters/skillmas'
  for f in common.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts:
    q=adapters/f.relative_to(common);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q)
  for b in ['gaia','masbench','macbench']:shutil.copyfile(REPO/f'benchmarks/{b}/adapters/skillmas/{b}.py',adapters/f'{b}.py')
  # Replace upstream connection configuration with a credential-free experiment configuration.
  model={'deepseek-flash':{'api_key':'${OPENAI_API_KEY}','temperature':0.2,'max_tokens':32768,'input_per_1m':0.20,'output_per_1m':1.20}}
  (stage/'skill_mas/model_config.json').write_text(json.dumps(model,indent=2))
  hashes={str(f.relative_to(stage)):hashlib.sha256(f.read_bytes()).hexdigest() for f in stage.rglob('*.py') if '.git' not in f.parts}
  (stage/'REPRODUCTION.json').write_text(json.dumps({'upstream':lock,'source_sha256':hashes},indent=2))
  shutil.move(str(stage),str(dst))
 print(dst)
if __name__=='__main__':main()
