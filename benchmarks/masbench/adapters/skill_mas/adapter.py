"""MASBench data/environment bridge. SkillMAS retains planning, execution and evolution."""
from __future__ import annotations
import asyncio,importlib.util,json,os,re,subprocess,sys,threading,select
from pathlib import Path
from benchmarks.common.skill_mas.base import NativeBenchmarkAdapter,NativeTask,NativeEvalResult,ROOT

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class GameRPC:
    def __init__(self,source,path,out):
        out.mkdir(parents=True,exist_ok=True)
        self.log=(out/'worker.log').open('a');self.lock=threading.Lock()
        self.p=subprocess.Popen([sys.executable,str(Path(__file__).parent/'masbench_support/game_worker.py'),str(source),str(path),str(out)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:self.actors=self.read()['actors']
        except BaseException:self.close();raise
    def read(self):
        if not select.select([self.p.stdout],[],[],90)[0]:raise TimeoutError('MASBench game RPC deadline')
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('MASBench game worker exited; inspect worker.log')
        obj=json.loads(line)
        if not obj.get('ok'):raise ValueError(obj.get('error'))
        return obj
    def call(self,op,**kwargs):
        with self.lock:
            self.p.stdin.write(json.dumps({'op':op,**kwargs})+'\n');self.p.stdin.flush();return self.read()['result']
    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
        for stream in (self.p.stdin,self.p.stdout):
            if stream:stream.close()
        self.log.close()

class MASBenchAdapter(NativeBenchmarkAdapter):
    name='masbench'
    def default_source(self):
        configured=os.getenv('MASBENCH_ROOT','').strip()
        if not configured:raise RuntimeError('Set MASBENCH_ROOT to the separately installed MASBench checkout.')
        return Path(configured).expanduser().resolve()
    def load_tasks(self,source=None,*,split='test',max_tasks=0):
        if split not in {'train','test'}:raise ValueError('MASBench split must be train or test')
        directory=Path(source or os.getenv('SKILL_MAS_MANIFEST_DIR',ROOT/'data/manifests/skill_mas/masbench')).expanduser().resolve();manifest=directory/f'{split}_manifest.json'
        selected=json.loads(manifest.read_text());root=self.default_source()
        reasoning={r['id']:r for r in json.loads((root/'Reasoning/reasoning.json').read_text())};tasks=[]
        for cat in ('reasoning','scheduling','game'):
            ids=selected[cat]['ids']
            if len(set(ids))!=len(ids):raise ValueError('duplicate manifest IDs')
            for tid in ids:
                raw={'category':cat,'id':tid,'source_root':str(root)}
                if cat=='reasoning':raw['row']=reasoning[tid];prompt=reasoning[tid]['question']
                elif cat=='scheduling':
                    a,b=tid.split('_');path=root/'Scheduling/meeting/minutes_10/5days'/a/b
                    if not (path/'task_plans').is_file():raise FileNotFoundError(path)
                    raw['path']=str(path);prompt='Find a shared continuous 20-minute meeting for five members across five days, using their private calendars.'
                else:
                    path=root/'Game/custom_map'/tid
                    if not path.is_file():raise FileNotFoundError(path)
                    raw['path']=str(path);prompt='Cooperate to explore, collect and deliver resources in a partially observed grid world, within 50 simultaneous environment rounds.'
                tasks.append(NativeTask(f'{cat}__{tid}',prompt,raw,split))
        return tasks[:max_tasks] if max_tasks else tasks
    def setup_runtime(self,task,context):
        cat=task.raw['category'];root=Path(task.raw['source_root']);context['mas_task']=task
        if cat=='reasoning':
            loader=load_module('_skillmas_mas_loader',root/'Reasoning/src/data_loader_v4_mas.py').V4MasDataLoader(str(root/'Reasoning/reasoning.json'),mode='context')
            sample=loader.sample(next(i for i,r in enumerate(loader._items) if r['id']==task.raw['id']))
            context['private']={'Alice':sample.context_first,'Bob':sample.context_second}
        elif cat=='scheduling':
            context['private']={f'Member{i}':json.loads((Path(task.raw['path'])/f'Member{i}_final_schedule.json').read_text()) for i in range(1,6)}
        else:
            rpc=GameRPC(root,Path(task.raw['path']),Path(context.get('artifact_dir',context['workspace']))/'game');context['game_rpc']=rpc;context['private']={n:None for n in rpc.actors}
        context['inboxes']={n:[] for n in context['private']};context['messages']=[]
    def build_prompt(self,task,context):
        actors=list(context['private']);cat=task.raw['category']
        fmt='FINAL ANSWER: <answer only>' if cat=='reasoning' else ('FINAL ANSWER: {"day":1,"start_minute":600,"end_minute":620}' if cat=='scheduling' else 'FINAL ANSWER: finished')
        return task.prompt+'\n'+f'Environment actor identities (use exact names or include exactly one identity as an underscore-delimited token in SubAgentRequest.role_name, e.g. collect_Alice_evidence): {json.dumps(actors)}. Each actor must run in native_tool mode. observe_private reveals ONLY the caller role\'s evidence/calendar/local view. Other roles have no private access. Use send_message and read_messages for explicit communication. Never read benchmark files or reference answers. Workflow planning and additional coordinator roles are allowed, but they have no private observations. '+('For Game: repeatedly invoke every actor once per environment round, up to 50 rounds. Each actor observes its local view, communicates only within range, and submits one game_action with move UP/DOWN/LEFT/RIGHT/STAY/PICK_UP/PUT_DOWN; return immediately after submitting, do not submit twice within one actor invocation. Physics advances only after all actors submit. Continue until done=true; do not stop after one pass. ' if cat=='game' else '')+'Return '+fmt
    def bind_actor(self,payload,context):
        role=str(payload.get('role_name','')) if isinstance(payload,dict) else ''
        matches=[name for name in context['private'] if re.search(r'(?<![a-z0-9])'+re.escape(name.lower())+r'(?![a-z0-9])',role.lower())]
        # Stage 2 names roles after subtasks (e.g. collect_alice_evidence).
        # Accept an explicit single actor token; ambiguous multi-actor names stay unprivileged.
        return matches[0] if len(matches)==1 else role
    def tool_schemas(self,task,context=None):
        def schema(n,d,p={},req=[]):return {'type':'function','function':{'name':n,'description':d,'parameters':{'type':'object','properties':p,'required':req,'additionalProperties':False}}}
        tools=[schema('observe_private','Observe only your own private evidence/calendar/local map; caller identity is bound by runtime.'),schema('read_messages','Read messages explicitly addressed to your role.'),schema('send_message','Send explicit text to a teammate; Game requires communication range.',{'recipient':{'type':'string'},'message':{'type':'string'}},['recipient','message'])]
        if task.raw['category']=='game':tools.append(schema('game_action','Submit your one action this round. World advances once all actors have acted.',{'move':{'type':'string','enum':['UP','DOWN','LEFT','RIGHT','STAY','PICK_UP','PUT_DOWN']}},['move']))
        return tools
    async def execute_scoped_tool(self,name,args,context,actor):
        if actor not in context['private']:raise PermissionError('Role has no environment actor identity; use exact registered role_name')
        rpc=context.get('game_rpc')
        if name=='observe_private':result=await asyncio.to_thread(rpc.call,'observe',actor=actor) if rpc else context['private'][actor]
        elif name=='read_messages':result=list(context['inboxes'][actor])
        elif name=='send_message':
            recipient=args['recipient']
            if recipient not in context['inboxes']:raise ValueError('unknown recipient')
            if rpc and not await asyncio.to_thread(rpc.call,'in_range',actor=actor,recipient=recipient):raise PermissionError('recipient outside communication range')
            row={'sender':actor,'recipient':recipient,'message':str(args['message'])};context['inboxes'][recipient].append(row);context['messages'].append(row);result={'delivered':True}
        elif name=='game_action' and rpc:result=await asyncio.to_thread(rpc.call,'action',actor=actor,action={'move':args['move']})
        else:raise ValueError('unavailable tool')
        return json.dumps(result,ensure_ascii=False,default=str)
    def evaluate(self,task,output,context):
        cat=task.raw['category'];answer=re.split(r'(?i)final\s*answer\s*:',output or '')[-1].strip();details={}
        if cat=='reasoning':
            m=load_module('_skillmas_mas_metrics',Path(task.raw['source_root'])/'Reasoning/src/metrics.py');score=m.cal_f1_score(str(task.raw['row']['answer']),answer);details={'prediction':answer,'reference':task.raw['row']['answer']}
        elif cat=='scheduling':
            score=0.0
            try:
                obj=json.loads(answer);day,start,end=[obj[k] for k in ('day','start_minute','end_minute')]
                if any(type(x) is not int for x in (day,start,end)):raise ValueError('integer values required')
                if not (1<=day<=5 and 0<=start<end<=1440 and end-start==20 and start%10==0):raise ValueError('invalid 20-minute slot')
                plans=json.loads((Path(task.raw['path'])/'task_plans').read_text())['meeting']['solutions'];s=(day-1)*144+start//10;e=(day-1)*144+end//10
                score=float(any(p['start_time']==s and p['end_time']==e for p in plans));details={'prediction':obj}
            except (ValueError,TypeError,KeyError) as exc:details={'invalid_prediction':str(exc)}
        else:details=context['game_rpc'].call('score');score=details['score']
        return NativeEvalResult(score==1.0,float(score),f'MASBench {cat}',details)
    def teardown_runtime(self,task,context):
        rpc=context.pop('game_rpc',None)
        if rpc:rpc.close()
