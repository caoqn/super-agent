"""Isolated JSON-lines RPC for upstream MASBench Game physics; no model calls."""
import sys,json,os,asyncio,traceback
from pathlib import Path
wire=os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
# Keep upstream print/redirection and module globals out of the training process.
sys.stdout=sys.stderr
class Brain:
    usage={}
    def system(self,*a): pass

def main():
    root,map_path,out=sys.argv[1:4]
    sys.path.insert(0,str(Path(root)/'Game'))
    os.chdir(out)
    from masbench.environment import MASEnvironment
    from masbench.agent import MASAgent
    from masbench.logger import MASLogger
    data=json.loads(Path(map_path).read_text());cfg=data['map'];ac=cfg.get('agent_config',{})
    names=[a.get('name',f'Agent{i+1}') for i,a in enumerate(cfg['agents'])]
    agents={n:MASAgent(n,Brain(),'Cooperate to collect and deliver resources.',enable_inventory=True,carrying_capacity=ac.get('item_number',4),view_memory=0,history_memory=0) for n in names}
    logger=MASLogger('skillmas',{},log_dir=str(Path(out)/'native_logs'))
    env=MASEnvironment('skillmas',agents,logger,task='Foraging',max_round=50,width=cfg['width'],height=cfg['height'],custom_map=data,view_size=ac.get('view_size',9),communication_size=ac.get('communication_size',15),message_phase_enabled=False,messages_enabled=True)
    pending={}
    def status():
        totals=env.tasks[env.task].count_items_on_tables(env);target=data['task']['scoring'];earned=sum(min(int(totals.get(k,0)),int(v)) for k,v in target.items());maximum=sum(int(v) for v in target.values())
        return {'round':env.round,'done':env.is_done(),'score':earned/maximum,'earned':earned,'maximum':maximum}
    wire.write(json.dumps({'ok':True,'actors':names})+'\n');wire.flush()
    for line in sys.stdin:
        try:
            req=json.loads(line);op=req['op'];name=req.get('actor');result=None
            if op=='observe': result={'round':env.round,'done':env.is_done(),'view':agents[name].gen_prompt(env.get_round_obs(agents[name]),{}),'pending':list(pending)}
            elif op=='action':
                if env.is_done(): raise ValueError('episode finished')
                if name in pending: raise ValueError('one action per actor per round; wait for teammates')
                action=req['action'];valid={'UP','DOWN','LEFT','RIGHT','STAY','PICK_UP','PUT_DOWN'}
                if action.get('move') not in valid: raise ValueError('invalid move')
                pending[name]=action
                if set(pending)==set(names):
                    for actor in names: asyncio.run(env.act(agents[actor],pending[actor]))
                    env.update();env.clear_round_obs_cache();pending.clear()
                result=status()
            elif op=='in_range':result=env.is_in_communication_range(name,req['recipient'])
            elif op=='score':result=status()
            elif op=='close':break
            else:raise ValueError('unknown operation')
            wire.write(json.dumps({'ok':True,'result':result},default=str)+'\n');wire.flush()
        except Exception as exc:
            wire.write(json.dumps({'ok':False,'error':str(exc)})+'\n');wire.flush()
    env.cleanup_output_logging()
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc(file=sys.stderr);raise
