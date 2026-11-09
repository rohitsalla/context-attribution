"""One new FP32 selection contract between frozen deletion and refitted selection."""
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
from .frontier_probe import flatten
from .v2.memory import Source
from .v2.run import raw_memory
from .v4.evidence import global_lexical


def rank_chunks(tokenizer,sources,question):
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
    ranked=sorted(candidates,key=lambda c:(-score(c),c[0].sid,c[1]))
    return [(s.sid,i,tuple(tokenizer.encode(f'\n[{s.sid}] {s.title}: {text}\n',add_special_tokens=False)))
            for s,i,text,_ in ranked]


def pack_fixed(ranked,source_order,budget,omit=None):
    kept=[c for c in ranked if c[0]!=omit];chosen=[];used=0
    for c in kept:
        if used+len(c[2])<=budget:chosen.append(c);used+=len(c[2])
    if not chosen and kept:
        sid,i,ids=kept[0];chosen=[(sid,i,ids[:budget])]
    order={sid:i for i,sid in enumerate(source_order)}
    return sorted(chosen,key=lambda c:(order[c[0]],c[1]))


def main():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .v2.hf import HFModel
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--study',choices=['hotpot_qwen7','hotpot_llama8','musique_qwen7','musique_llama8'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    benchmark,key=args.study.split('_');folder=args.root/'results/frontier'/args.study
    done=json.loads((folder/'completed.json').read_text());assert done['complete']
    trace=folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
    dynamic={r['item_id']:r for r in rows if r['method']=='global_512_recomputed'}
    frozen={r['item_id']:r for r in rows if r['method']=='global_512_frozen'}
    dataset=args.root/('data/v2-hotpot/dev.jsonl' if benchmark=='hotpot' else 'data/v4-musique/frontier32.jsonl')
    old_manifest=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==old_manifest['dataset_sha256']
    items=[json.loads(s) for s in dataset.read_text().splitlines()]
    assert len(items)==done['n'] and set(dynamic)==set(frozen)=={i['id'] for i in items}
    if args.limit is not None:
        assert 1<=args.limit<=len(items);items=items[:args.limit]
    assert all(i['split']=='dev' for i in items)
    info=json.loads((args.root/'research/v2/models.json').read_text())[key];paths=json.loads(args.paths.read_text())
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    model=HFModel(paths[key],info['revision'],model_id=info['model_id']);model.model.float().requires_grad_(False)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'adaptive development, fixed-ranking repacking versus two archived intervention contracts',
        'study':args.study,'runtime':model.runtime(),'n_expected':len(items),'budget':512,
        'reference_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_score_tolerance':1e-6,'repeat_score_tolerance':1e-7,
        'numerics':'FP32 weights, TF32 off, math attention, explicit 4D causal mask, double log-softmax',
        'qa_scope':'copied from identical factual token stream; no new answer generation',
        'limitation':'fixed-rank repacking changes content/length after deletion; not a matched-length or unique mediation contrast'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def score(prefix,target):
        ids=torch.tensor([list(prefix)+list(target)],device=model.device);length=ids.shape[-1]
        mask=torch.ones((length,length),device=model.device,dtype=torch.bool).tril()[None,None]
        with torch.inference_mode():
            logits=model.model(input_ids=ids,position_ids=torch.arange(length,device=model.device)[None],
                attention_mask=mask,use_cache=False).logits[0,len(prefix)-1:-1].double()
            return float(logits.log_softmax(-1).gather(-1,ids[0,len(prefix):,None]).mean())
    count=0;started=time.monotonic()
    with sdpa_kernel(SDPBackend.MATH),(args.output/'traces.jsonl').open('w') as handle:
        for item in items:
            ref=dynamic[item['id']];old_frozen=frozen[item['id']];target=ref['common_target_tokens']
            sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']];order=[s.sid for s in sources]
            ranked=rank_chunks(model.tokenizer,sources,item['question']);chunks=pack_fixed(ranked,order,512);memory=flatten(chunks)
            assert memory==global_lexical(model.tokenizer,sources,item['question'],512)
            assert hashlib.sha256(json.dumps(memory).encode()).hexdigest()==ref['factual_memory_sha256']==old_frozen['factual_memory_sha256']
            prefix=model.qa_ids(item['question'],memory);base=score(prefix,target)
            base_error=abs(base-ref['factual_target_logp'])
            raw_error=abs(score(model.qa_ids(item['question'],raw_memory(model.tokenizer,sources)),target)-ref['raw_factual_target_logp'])
            if max(base_error,raw_error)>1e-6:raise ValueError('Archived reference score drift')
            effects={};positions={};changed_flags={};repeat_error=0.
            chosen=min(order,key=lambda sid:hashlib.sha256((item['id']+':'+sid).encode()).hexdigest())
            for source in sources:
                changed=flatten(pack_fixed(ranked,order,512,source.sid));assert len(changed)<=512
                changed_prefix=model.qa_ids(item['question'],changed);value=score(changed_prefix,target)
                effects[source.sid]=base-value;positions[source.sid]=len(changed)
                if source.sid==chosen:repeat_error=abs(value-score(changed_prefix,target))
                recomputed=global_lexical(model.tokenizer,[s for s in sources if s.sid!=source.sid],item['question'],512)
                changed_flags[source.sid]={'fixed_repack_vs_frozen_tokens_differ':changed!=flatten(chunks,source.sid),
                    'refitted_vs_fixed_repack_tokens_differ':recomputed!=changed}
            if repeat_error>1e-7:raise ValueError('Repeatability check failed')
            row={'item_id':item['id'],'method':'global_512_fixed_rank_refill','qa':ref['qa'],'answer':ref['answer'],
                'common_target_tokens':target,'raw_effects':ref['raw_effects'],'effects':effects,
                'frozen_effects':old_frozen['effects'],'recomputed_effects':ref['effects'],
                'memory_positions':len(memory),'intervention_positions':positions,'token_change_checks':changed_flags,
                'factual_memory_sha256':ref['factual_memory_sha256'],'factual_target_logp':base,
                'reference_factual_mean_logp_error':base_error,'reference_raw_mean_logp_error':raw_error,
                'repeat_mean_logp_error':repeat_error,'reference_qa_scope':manifest['qa_scope']}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());count+=1
            print(item['id'],'complete',count,'seconds',round(time.monotonic()-started,1),flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'n':count,'elapsed_seconds':time.monotonic()-started},indent=2)+'\n')


if __name__=='__main__':main()
