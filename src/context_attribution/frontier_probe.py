"""FP32 development frontier with explicit provenance and selection controls."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .attribution import sentences
from .v2.memory import Source
from .v2.run import raw_memory
from .v4.evidence import answer_quality, global_lexical


def selected_chunks(tokenizer, sources, question, budget):
    """Same selection as the frozen global baseline, retaining chunk ownership."""
    candidates=[]
    for source in sources:
        for index,text in enumerate(sentences(source.text) or [source.text]):
            candidates.append((source,index,text,Counter(re.findall(r'\w+',(source.title+' '+text).lower()))))
    query=set(re.findall(r'\w+',question.lower()));n=len(candidates)
    avg=sum(sum(b.values()) for *_,b in candidates)/max(n,1)
    df={w:sum(w in c[3] for c in candidates) for w in query}
    def score(c):
        bag=c[3];length=sum(bag.values())
        return sum(math.log(1+(n-df[w]+.5)/(df[w]+.5))*bag[w]*2.2/
                   (bag[w]+1.2*(.25+.75*length/max(avg,1))) for w in query if bag[w])
    ranked=sorted(candidates,key=lambda c:(-score(c),c[0].sid,c[1]));chosen=[];used=0
    for source,index,text,_ in ranked:
        ids=tuple(tokenizer.encode(f'\n[{source.sid}] {source.title}: {text}\n',add_special_tokens=False))
        if used+len(ids)<=budget:
            chosen.append((source.sid,index,ids));used+=len(ids)
    if not chosen and ranked:
        s,i,text,_=ranked[0]
        chosen=[(s.sid,i,tuple(tokenizer.encode(f'\n[{s.sid}] {s.title}: {text}\n',add_special_tokens=False)[:budget]))]
    order={s.sid:i for i,s in enumerate(sources)}
    return sorted(chosen,key=lambda c:(order[c[0]],c[1]))


def flatten(chunks, omit=None):
    return tuple(t for sid,_,tokens in chunks if sid!=omit for t in tokens)


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .v2.hf import HFModel
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model-key',choices=['qwen7','llama8'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=96)
    args=p.parse_args()
    if not 1<=args.limit<=96:raise ValueError('Development limit must be 1..96')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    items=[json.loads(s) for s in args.dataset.read_text().splitlines()][:args.limit]
    if not items or any(i['split']!='dev' for i in items):raise ValueError('Development only')
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    registry=json.loads((args.root/'research/v2/models.json').read_text());info=registry[args.model_key]
    paths=json.loads(args.paths.read_text())
    model=HFModel(paths[args.model_key],info['revision'],model_id=info['model_id'])
    model.model.float().requires_grad_(False);model.model.generation_config.repetition_penalty=1.
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'adaptive development, new FP32 answering/scoring distribution; no positional equivalence claim',
        'runtime':model.runtime(),'budgets':[128,256,512,1024],'frozen_selection_budget':512,
        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'limit':args.limit,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helpers_sha256':{str(f.relative_to(args.root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [args.root/'src/context_attribution'/s for s in ['v2/hf.py','v4/evidence.py']]},
        'attention':'forced math; explicit 4D causal boolean mask for all likelihood scores; explicit 2D ones for greedy generation',
        'scoring':'mean target log probability, double-precision log-softmax, TF32 disabled',
        'target':'fresh FP32 raw factual greedy answer; fixed across all methods/interventions per question',
        'selection_control':'same factual 512 tokens/answer; frozen removes owned chunks, recomputed refits IDF and repacks',
        'generation_cap':64,'generation_repetition_penalty':1.,'python_hash_seed':os.environ.get('PYTHONHASHSEED')}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def score(prefix,target):
        ids=torch.tensor([list(prefix)+list(target)],device=model.device)
        length=ids.shape[-1];mask=torch.ones((length,length),device=model.device,dtype=torch.bool).tril()[None,None]
        with torch.inference_mode():
            logits=model.model(input_ids=ids,position_ids=torch.arange(length,device=model.device)[None],
                               attention_mask=mask,use_cache=False).logits[0,len(prefix)-1:-1].double()
            return float(logits.log_softmax(-1).gather(-1,ids[0,len(prefix):,None]).mean())
    started=time.monotonic();count=0
    with sdpa_kernel(SDPBackend.MATH),(args.output/'traces.jsonl').open('w') as handle:
        for item in items:
            sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
            raw=tuple(raw_memory(model.tokenizer,sources));raw_prefix=model.qa_ids(item['question'],raw)
            raw_answer=model.generate(raw_prefix);target=raw_answer['tokens']
            if not target:raise ValueError('Empty raw target')
            raw_base=score(raw_prefix,target)
            repeat_error=abs(raw_base-score(raw_prefix,target))
            if repeat_error>1e-7:raise ValueError('Same-shape factual scoring is not repeatable')
            def changed_raw(sid):
                return raw_memory(model.tokenizer,[s for s in sources if s.sid!=sid])
            raw_effects={s.sid:raw_base-score(model.qa_ids(item['question'],changed_raw(s.sid)),target) for s in sources}
            common={'item_id':item['id'],'hops':item.get('hops'),'raw_qa':answer_quality(raw_answer['text'],item),
                    'raw_answer':raw_answer,'common_target_tokens':target,'raw_effects':raw_effects,
                    'raw_factual_target_logp':raw_base,
                    'same_shape_repeat_error':repeat_error,'raw_memory_positions':len(raw)}
            rows=[dict(common,method='raw',qa=common['raw_qa'],answer=raw_answer,memory_positions=len(raw),
                       effects=raw_effects,factual_target_logp=raw_base,raw_identity_endpoint_error=0.)]
            for budget in [128,256,512,1024]:
                chunks=selected_chunks(model.tokenizer,sources,item['question'],budget);tokens=flatten(chunks)
                if tokens!=global_lexical(model.tokenizer,sources,item['question'],budget):raise ValueError('Selector drift')
                if len(tokens)>budget:raise ValueError('Budget exceeded')
                prefix=model.qa_ids(item['question'],tokens);answer=model.generate(prefix);base=score(prefix,target)
                for frozen in ([False,True] if budget==512 else [False]):
                    effects={};positions={};replacement={}
                    for source in sources:
                        kept=[s for s in sources if s.sid!=source.sid]
                        changed=flatten(chunks,source.sid) if frozen else flatten(selected_chunks(model.tokenizer,kept,item['question'],budget))
                        if len(changed)>budget:raise ValueError('Intervention budget exceeded')
                        if not frozen and changed!=global_lexical(model.tokenizer,kept,item['question'],budget):raise ValueError('Deleted selector drift')
                        effects[source.sid]=base-score(model.qa_ids(item['question'],changed),target)
                        positions[source.sid]=len(changed)
                        replacement[source.sid]=changed!=flatten(chunks,source.sid)
                    rows.append(dict(common,method=f'global_{budget}'+('_frozen' if frozen else '_recomputed'),
                        qa=answer_quality(answer['text'],item),answer=answer,memory_positions=len(tokens),budget=budget,
                        effects=effects,factual_target_logp=base,intervention_positions=positions,selection_changed_after_deletion=replacement,
                        factual_memory_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        source_positions={s.sid:sum(len(c[2]) for c in chunks if c[0]==s.sid) for s in sources}))
            # Passthrough endpoint is re-executed, not a copied zero-error metric.
            endpoint_base=score(model.qa_ids(item['question'],tuple(raw_memory(model.tokenizer,sources))),target)
            endpoint={s.sid:endpoint_base-
                score(model.qa_ids(item['question'],tuple(changed_raw(s.sid))),target) for s in sources}
            endpoint_error=max(abs(endpoint[k]-raw_effects[k]) for k in endpoint)
            if endpoint_error>1e-7:raise ValueError('Raw passthrough endpoint failed')
            rows[0]['raw_identity_endpoint_error']=endpoint_error
            # Whole-question writes: a timeout leaves incomplete status and inspectable completed questions.
            for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
            handle.flush();os.fsync(handle.fileno());count+=1
            print(item['id'],'complete',count,'seconds',round(time.monotonic()-started,1),flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'n':count,'conditions':count*6,
        'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')


if __name__=='__main__':main()
