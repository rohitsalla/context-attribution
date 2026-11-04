"""Measure actual selective-replay execution without full-reference lookups."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from .backend import HFBackend, answer_prompt
from .experiment import branch,answer_correct
from .attribution import recursive,lexical,tracing,select


def run_online(backend,item,cap=128,seed=1729):
    original=branch(backend,item,item['sources'],cap,seed=seed)
    answer=original['fixed_answer'];methods=[]
    scorers=[('recursive_surrogate',lambda:recursive(backend,item,original['summary'],answer,32,.01,seed)),
             ('lexical',lambda:lexical(item,answer)),('source_id',lambda:tracing(item,original['summary'])),
             ('random',lambda:None)]
    for name,scorer in scorers:
        start=len(backend.records);tick=time.perf_counter();scores=scorer()
        if scores is None:
            ranked=sorted(s['id'] for s in item['sources']);random.Random(seed).shuffle(ranked)
        else: ranked=select(scores,3,seed)
        verified={s['id']:None for s in item['sources']};budgets=[]
        for budget,sid in enumerate(ranked[:3],1):
            trial=branch(backend,item,[s for s in item['sources'] if s['id']!=sid],cap,answer,seed)
            verified[sid]=original['logprob']-trial['logprob']
            budgets.append({'budget':budget,'selected':ranked[:budget],
                            'verified_effects':dict(verified),'untested':[k for k,v in verified.items() if v is None],
                            'cost':backend.cost(start),'end_to_end_seconds':time.perf_counter()-tick,
                            'latest_replay':trial})
        methods.append({'method':name,'candidate_scores':scores,'budgets':budgets})
    return {'item_id':item['id'],'base_id':item['base_id'],'cap':cap,'original':original,
            'correct':answer_correct(item,answer),'methods':methods,'reference_lookups_used':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=[json.loads(s) for s in args.dataset.read_text().splitlines()]
    selected=[data[i] for i in [0,10,20,30,40,50,60,70,80,90,100,110]]
    args.output.mkdir(parents=True,exist_ok=True)
    fingerprint={'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                 'model':args.model,'revision':args.revision,
                 'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
    manifest=args.output/'manifest.json';trace=args.output/'traces.jsonl'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprint: raise ValueError('Online resume mismatch')
    rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
    completed={r['item_id'] for r in rows}
    if all(r['id'] in completed for r in selected): print('Online subset already complete');return
    backend=HFBackend(args.model,'cuda',args.revision)
    manifest.write_text(json.dumps(fingerprint,indent=2))
    with trace.open('a') as handle:
        for item in selected:
            if item['id'] in completed:continue
            row=run_online(backend,item)
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print('online',item['id'],'complete',flush=True)


if __name__=='__main__':main()
