"""Post-hoc numerical diagnostics on selected positional-equivalence failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from context_attribution.v2.hf import HFModel
from context_attribution.position_probe import remove_layout,score_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--case-limit',type=int);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh output required')
    prior=args.root/'results/position-probe-v2/traces.jsonl';old=[json.loads(s) for s in prior.read_text().splitlines()]
    worst=sorted(old,key=lambda r:-max(c['dense_vs_sparse_mean_logprob_error'] for c in r['crosschecks']))[:6]
    backend=sorted(old,key=lambda r:-max(c['default_vs_math_mean_logprob_error'] for c in r['crosschecks']))[:6]
    best=sorted(old,key=lambda r:max(c['dense_vs_sparse_mean_logprob_error'] for c in r['crosschecks']))[:4]
    chosen={r['item_id']:r for r in worst+backend+best}
    if args.case_limit is not None:
        if args.case_limit<1:raise ValueError('Positive case limit required')
        chosen=dict(list(chosen.items())[:args.case_limit])
    items={r['id']:r for r in map(json.loads,(args.root/'data/v2-hotpot/dev.jsonl').read_text().splitlines())}
    refs={r['item_id']:r for r in map(json.loads,(args.root/'results/v4/evidence_hotpot_qwen7/traces.jsonl').read_text().splitlines()) if r['method']=='raw'}
    paths=json.loads(args.paths.read_text());info=json.loads((args.root/'research/v2/models.json').read_text())['qwen7']
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    model=HFModel(paths['qwen7'],info['revision'],model_id=info['model_id'],device=args.device);model.model.requires_grad_(False)
    if args.device=='cpu':torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'post-hoc selected numerical failures and low-error controls; not population inference',
              'selection':'six largest dense/sparse errors, six largest backend errors, four smallest dense/sparse errors; union',
              'case_ids':list(chosen),'native_runtime':model.runtime(),'tf32':False,'float32_matmul_precision':'highest',
              'device':args.device,'case_limit':args.case_limit,'cpu_scope':'CPU validation does not certify CUDA behavior' if args.device=='cpu' else None,
              'torch_num_threads':torch.get_num_threads(),
              'prior_trace_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
              'reference_sha256':hashlib.sha256((args.root/'results/v4/evidence_hotpot_qwen7/traces.jsonl').read_bytes()).hexdigest(),
              'mask_helper_sha256':hashlib.sha256((args.root/'src/context_attribution/position_probe.py').read_bytes()).hexdigest(),
              'score_accumulation':'float64 log-softmax in both model precisions',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));cases={};measurements={}
    for qid,oldrow in chosen.items():
        item=items[qid];left,right=model.qa_sides(item['question']);prefix=list(left);spans={}
        for s in item['sources']:
            start=len(prefix);prefix+=model.tokenizer.encode(f"\n[{s['id']}] {s['title']}: {s['text']}\n",add_special_tokens=False)
            spans[s['id']]=(start,len(prefix))
        prefix+=right;target=refs[qid]['common_target_tokens'];full=prefix+target
        cases[qid]=(full,len(target),{c['source']:spans[c['source']] for c in oldrow['crosschecks']})
    def score(tokens,positions,mask,n,explicit4d):
        ids,pos,attention=score_inputs(tokens,positions,mask,model.device)
        if explicit4d:
            live=attention.bool()[0];causal=torch.ones((len(tokens),len(tokens)),device=model.device,dtype=torch.bool).tril()
            attention=(causal&live[None,:])[None,None,:,:]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            logits=model.model(input_ids=ids,position_ids=pos,attention_mask=attention,use_cache=False).logits[0,-n-1:-1].double()
            return float(logits.log_softmax(-1).gather(-1,ids[0,-n:,None]).mean())
    precisions=['float32'] if args.device=='cpu' else ['bfloat16','float32']
    for precision in precisions:
        if precision=='float32':model.model.float()
        for qid,(full,n,spans) in cases.items():
            for sid,span in spans.items():
                tokens,positions,mask=remove_layout(full,span);scores={}
                for explicit4d in [False,True]:
                    suffix='4d' if explicit4d else '2d'
                    scores['sparse_'+suffix]=score(tokens,positions,None,n,explicit4d)
                    scores['dense_'+suffix]=score(full,list(range(len(full))),mask,n,explicit4d)
                scores['compact_4d']=score(tokens,list(range(len(tokens))),None,n,True)
                row={'item_id':qid,'source':sid,'precision':precision,'scores':scores,
                     'dense_sparse_2d_error':abs(scores['dense_2d']-scores['sparse_2d']),
                     'dense_sparse_4d_error':abs(scores['dense_4d']-scores['sparse_4d']),
                     'sparse_2d_4d_error':abs(scores['sparse_2d']-scores['sparse_4d']),
                     'dense_2d_4d_error':abs(scores['dense_2d']-scores['dense_4d']),
                     'position_protocol_gap_4d':scores['sparse_4d']-scores['compact_4d']}
                measurements[(qid,sid,precision)]=row
                with (args.output/'traces.jsonl').open('a') as handle:
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(precision,qid,'complete',flush=True)
    result={'complete':True,'selected_questions':len(cases),'scope':manifest['scope'],'by_precision':{}}
    for precision in precisions:
        values=[r for r in measurements.values() if r['precision']==precision]
        result['by_precision'][precision]={key:max(r[key] for r in values) for key in ['dense_sparse_2d_error','dense_sparse_4d_error','sparse_2d_4d_error','dense_2d_4d_error']}
        result['by_precision'][precision]['all_4d_checks_within_1e_minus4']=all(r['dense_sparse_4d_error']<=1e-4 for r in values)
    result['final_runtime']=model.runtime()
    (args.output/'completed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
