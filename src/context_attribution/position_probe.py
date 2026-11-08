"""Declared deletion-protocol sensitivity on fixed, completed raw-answer spans.

Retaining position IDs defines a different intervention; it is not labeled a
correction or a uniquely true source effect. No generation or test access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from .v2.hf import HFModel
from .v2.memory import digest


def remove_layout(tokens,span):
    start,end=span
    if not 0<=start<end<=len(tokens):raise ValueError('Invalid intervention span')
    keep=[i for i in range(len(tokens)) if not start<=i<end]
    return [tokens[i] for i in keep],keep,[int(not start<=i<end) for i in range(len(tokens))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh probe output required')
    items=[json.loads(s) for s in (args.root/'data/v2-hotpot/dev.jsonl').read_text().splitlines()]
    if any(i['split']!='dev' for i in items):raise ValueError('Development only')
    refs={r['item_id']:r for r in map(json.loads,args.reference.read_text().splitlines()) if r['method']=='raw'}
    if set(refs)!={i['id'] for i in items}:raise ValueError('Complete matched reference required')
    paths=json.loads(args.paths.read_text());info=json.loads((args.root/'research/v2/models.json').read_text())['qwen7']
    model=HFModel(paths['qwen7'],info['revision'],model_id=info['model_id'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'manifest.json').write_text(json.dumps({'runtime':model.runtime(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256((args.root/'data/v2-hotpot/dev.jsonl').read_bytes()).hexdigest(),
        'scope':'deletion protocol sensitivity; alternative estimands, not causal credit ground truth',
        'dense_crosscheck_mean_logprob_tolerance':.001},indent=2))
    def score(tokens,positions,mask=None,target_length=None):
        ids=torch.tensor([tokens],device=model.device);pos=torch.tensor([positions],device=model.device)
        attention=None if mask is None else torch.tensor([mask],device=model.device)
        with torch.inference_mode():
            prefix=len(tokens)-target_length
            logits=model.model(input_ids=ids,position_ids=pos,attention_mask=attention,use_cache=False).logits[0,prefix-1:-1].float()
            return float(logits.log_softmax(-1).gather(-1,ids[0,prefix:,None]).sum())
    all_rows=[]
    with (args.output/'traces.jsonl').open('w') as handle:
        for item in items:
            ref=refs[item['id']];target=ref['common_target_tokens'];n=len(target)
            left,right=model.qa_sides(item['question']);prefix=list(left);spans={}
            for source in item['sources']:
                start=len(prefix);prefix+=model.tokenizer.encode(f"\n[{source['id']}] {source['title']}: {source['text']}\n",add_special_tokens=False)
                spans[source['id']]=(start,len(prefix))
            prefix+=right;full=prefix+target
            base=score(full,list(range(len(full))),target_length=n)
            original=model.score(prefix,target)
            if abs(base-original)/n>.001:raise ValueError('Position-aware scoring failed factual wrapper control')
            compact={};fixed={};cross=[]
            selected=sorted(spans,key=lambda sid:digest([item['id'],sid,'position-crosscheck']))[:2]
            for sid,span in spans.items():
                tokens,positions,mask=remove_layout(full,span)
                compact[sid]=(base-score(tokens,list(range(len(tokens))),target_length=n))/n
                preserved=score(tokens,positions,target_length=n);fixed[sid]=(base-preserved)/n
                if sid in selected:
                    dense=score(full,list(range(len(full))),mask,target_length=n)
                    cross.append({'source':sid,'dense_vs_sparse_mean_logprob_error':abs(dense-preserved)/n,
                                  'within_tolerance':abs(dense-preserved)/n<=.001})
            row={'item_id':item['id'],'raw_qa':ref['qa'],'compact_position_effects':compact,'retained_position_effects':fixed,
                'support_ids':item['support_ids'],'crosschecks':cross,'factual_wrapper_mean_logprob_error':abs(base-original)/n,
                'max_previous_deletion_effect_drift':max(abs(compact[s]-ref['effects'][s]) for s in compact),
                'note':'Effects are mean log p on the same fixed raw target; positional gaps include continuation positions.'}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno());all_rows.append(row)
            print(item['id'],'complete',flush=True)
    result={'complete':True,'n':len(all_rows),'scope':'one model/benchmark protocol diagnostic; no novelty/general mechanism claim'}
    for name in ['compact_position_effects','retained_position_effects']:
        scale=sum(abs(v) for r in all_rows for v in r[name].values())
        outside=sum(abs(v) for r in all_rows for k,v in r[name].items() if k not in r['support_ids'])
        result[name]={'pooled_l1':scale,'fraction_outside_annotated_support':outside/max(scale,1e-8)}
    result['max_dense_sparse_mean_logprob_error']=max(c['dense_vs_sparse_mean_logprob_error'] for r in all_rows for c in r['crosschecks'])
    result['all_dense_sparse_crosschecks_within_tolerance']=all(c['within_tolerance'] for r in all_rows for c in r['crosschecks'])
    result['max_previous_effect_drift']=max(r['max_previous_deletion_effect_drift'] for r in all_rows)
    (args.output/'completed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
