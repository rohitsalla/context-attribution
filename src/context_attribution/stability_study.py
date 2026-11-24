"""Prospective effect-rescaling and budget study; shared raw targets and checkpoints."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
from .generalization import atomic_text, omissions, POLICIES
from .heldout_followup import digest
from .frontier_probe import flatten, selected_chunks
from .selection_mechanism import pack_fixed, rank_chunks
from .recomp_selector import MODEL_ID, REVISION, RecompSelector
from .v2.memory import Source
from .v2.run import raw_memory
from .v4.evidence import answer_quality

MODELS = ('qwen7', 'mistral7')
DATASETS = ('musique', '2wiki')
BUDGETS = (256, 512, 1024, 2048)
SHARDS = 8


def read_cohort(project, dataset):
    manifest = json.loads((project/'data/stability/manifest.json').read_text())
    path = project/'data/stability'/f'{dataset}.jsonl'
    assert digest(path.read_bytes()) == manifest['files'][dataset]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == sum(manifest['counts'][dataset].values())
    for split, count in manifest['counts'][dataset].items(): assert sum(r['split']==split for r in rows) == count
    return rows, manifest


def chosen_rows(rows, shard=None, smoke=False):
    if smoke:
        test = [r for r in rows if r['split']=='test']
        first = min(test, key=lambda r: digest(('stability-smoke-v1|'+r['id']).encode()))
        longest = max(test, key=lambda r: sum(len(s['text']) for s in r['sources']))
        return [first] if first['id']==longest['id'] else [first,longest]
    return [r for split in ('dev','test') for r in [r for r in rows if r['split']==split][shard::SHARDS]]


def coverage(chunks, source_ids):
    return {sid: sum(len(c[2]) for c in chunks if c[0]==sid) for sid in source_ids}


def donor_replacement(factual, repacked, donor_tokens, offset):
    """Replace newly admitted chunks only, preserving exact branch length/order."""
    if not donor_tokens: raise ValueError('Empty donor stream')
    original = set(factual); changed = []; added = 0
    for sid,index,tokens in repacked:
        if (sid,index,tokens) in original: changed.extend(tokens)
        else:
            changed.extend(donor_tokens[(offset+added+i)%len(donor_tokens)] for i in range(len(tokens)))
            added += len(tokens)
    assert len(changed) == sum(len(c[2]) for c in repacked)
    return tuple(changed), added


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--recomp-path',type=Path,required=True);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--dataset',choices=DATASETS);p.add_argument('--shard',type=int,choices=range(SHARDS))
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.smoke == (a.dataset is None and a.shard is None)
    if not a.smoke: assert a.dataset is not None and a.shard is not None
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)


def run(a):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .v2.hf import HFModel
    selected=[];cohorts={};data_manifests={}
    for dataset in DATASETS if a.smoke else (a.dataset,):
        rows,data_manifest=read_cohort(a.project,dataset);data_manifests[dataset]=data_manifest;cohorts[dataset]={r['id']:r for r in rows}
        selected.extend((dataset,r) for r in chosen_rows(rows,a.shard,a.smoke))
    registry=json.loads((a.project/'research/v4/GENERALIZATION_MODELS.json').read_text());info=registry[a.model]
    paths=json.loads(a.paths.read_text());torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    reader=HFModel(paths[a.model],info['revision'],model_id=info['model_id']);reader.model.float().requires_grad_(False)
    reader.model.generation_config.repetition_penalty=1.
    selector=RecompSelector(a.recomp_path,reader.device)
    scientific_files=['src/context_attribution/stability_study.py','src/context_attribution/selection_mechanism.py',
        'src/context_attribution/frontier_probe.py','src/context_attribution/recomp_selector.py','src/context_attribution/v2/hf.py',
        'src/context_attribution/v2/run.py','src/context_attribution/v2/memory.py','src/context_attribution/generalization.py',
        'src/context_attribution/attribution.py','src/context_attribution/v4/evidence.py','review_stability.py','prepare_stability.py','research/v5/STABILITY_PROTOCOL.md','research/v4/GENERALIZATION_MODELS.json']
    receipt={'model':a.model,'dataset':a.dataset,'shard':a.shard,'smoke':a.smoke,'runtime':reader.runtime(),
        'source_hashes':{f:digest((a.project/f).read_bytes()) for f in scientific_files},
        'data_manifest_sha256':digest((a.project/'data/stability/manifest.json').read_bytes()),
        'recomp':MODEL_ID,'recomp_revision':REVISION,'budgets':BUDGETS,'policies':POLICIES,'expected_questions':len(selected),
        'python_hash_seed':os.environ.get('PYTHONHASHSEED'),'donor_control':'cross-question donor, same split; exact replacement length, not inert padding'}
    receipt=json.loads(json.dumps(receipt));mp=a.output/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=receipt:raise ValueError('Resume identity mismatch')
    else:atomic_text(mp,json.dumps(receipt,indent=2)+'\n')
    trace=a.output/'traces.jsonl';saved=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
    expected=[(d,r['id']) for d,r in selected]
    assert len(saved)<=len(expected) and [(r['dataset'],r['item_id']) for r in saved]==expected[:len(saved)]
    def direct(question,memory,target):
        prefix=reader.qa_ids(question,memory);ids=torch.tensor([prefix+list(target)],device=reader.device);n=ids.shape[-1]
        mask=torch.ones((n,n),dtype=torch.bool,device=reader.device).tril()[None,None]
        with torch.inference_mode():
            logits=reader.model(input_ids=ids,position_ids=torch.arange(n,device=reader.device)[None],
                 attention_mask=mask,use_cache=False,logits_to_keep=len(target)+1).logits[0,:-1].double()
            assert logits.shape[0]==len(target)
            value=float(logits.log_softmax(-1).gather(-1,ids[0,-len(target):,None]).mean())
            if not math.isfinite(value):raise ValueError('Nonfinite score')
            return value
    started=time.monotonic()
    with sdpa_kernel(SDPBackend.MATH):
        for dataset,item in selected[len(saved):]:
            question=item['question'];sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']];order=[s.sid for s in sources]
            raw=raw_memory(reader.tokenizer,sources);answer=reader.generate(reader.qa_ids(question,raw));target=answer['tokens']
            if not target:raise ValueError('Empty raw greedy target: '+item['id'])
            cache={}
            def score(memory):
                memory=tuple(memory)
                if memory not in cache:cache[memory]=direct(question,memory,target)
                return cache[memory]
            raw_base=score(raw);repeat_error=abs(raw_base-direct(question,raw,target));assert repeat_error<=1e-7
            plans=omissions(item);raw_branches={','.join(o):raw_memory(reader.tokenizer,[s for s in sources if s.sid not in o]) for o in plans}
            raw_effects={k:raw_base-score(v) for k,v in raw_branches.items()}
            sampled=min(order,key=lambda sid:digest((item['id']+'|'+sid).encode()))
            endpoint_error=abs((raw_base-direct(question,raw_branches[sampled],target))-raw_effects[sampled]);assert endpoint_error<=1e-7
            bm25_rank=rank_chunks(reader.tokenizer,sources,question);recomp_rank=selector.rank(reader.tokenizer,sources,question)
            donor_id=data_manifests[dataset]['donor_map'][dataset][item['id']];donor=cohorts[dataset][donor_id];assert donor['split']==item['split']
            donor_tokens=raw_memory(reader.tokenizer,[Source(s['id'],s['title'],s['text']) for s in donor['sources']])
            results={}
            for budget in BUDGETS:
                factual={'bm25':pack_fixed(bm25_rank,order,budget),'recomp':selector.pack(recomp_rank,sources,budget)}
                assert flatten(factual['bm25'])==flatten(selected_chunks(reader.tokenizer,sources,question,budget))
                bases={s:score(flatten(chunks)) for s,chunks in factual.items()}
                answers={s:reader.generate(reader.qa_ids(question,flatten(chunks))) for s,chunks in factual.items()}
                effects={s:{} for s in POLICIES};lengths={s:{} for s in POLICIES};changes={s:{} for s in ('bm25','recomp')}
                for omit in plans:
                    key=','.join(omit);kept=[s for s in sources if s.sid not in omit]
                    fixed=pack_fixed([c for c in bm25_rank if c[0] not in omit],order,budget)
                    repacked=pack_fixed([c for c in recomp_rank if c[0] not in omit],order,budget)
                    memories={'bm25_frozen':flatten([c for c in factual['bm25'] if c[0] not in omit]),
                        'bm25_fixed_rank':flatten(fixed),'bm25_refitted':flatten(selected_chunks(reader.tokenizer,kept,question,budget)),
                        'recomp_frozen':flatten([c for c in factual['recomp'] if c[0] not in omit]),'recomp_repacked':flatten(repacked)}
                    for policy,memory in memories.items():
                        effects[policy][key]=bases[policy.split('_')[0]]-score(memory);lengths[policy][key]=len(memory);assert len(memory)<=budget
                    for s,branch in [('bm25',fixed),('recomp',repacked)]:
                        original=set(factual[s]);branchset=set(branch)
                        changes[s][key]={'new_token_count':sum(len(c[2]) for c in branch if c not in original),
                            'displaced_surviving_tokens':sum(len(c[2]) for c in factual[s] if c[0] not in omit and c not in branchset)}
                controls={}
                if budget==512:
                    for s,ranking in [('bm25',bm25_rank),('recomp',recomp_rank)]:
                        retained=sorted({c[0] for c in factual[s]});sid=min(retained,key=lambda sid:digest(('length-control-v1|'+item['id']+'|'+s+'|'+sid).encode()))
                        branch=pack_fixed([c for c in ranking if c[0]!=sid],order,budget)
                        donor_memory,added=donor_replacement(factual[s],branch,donor_tokens,int(digest((item['id']+'|'+s).encode())[:12],16))
                        actual=flatten(branch);assert len(actual)==len(donor_memory)
                        controls[s]={'source':sid,'donor_item':donor_id,'replacement_tokens':added,'branch_length':len(actual),
                            'actual_effect':bases[s]-score(actual),'donor_effect':bases[s]-score(donor_memory),
                            'actual_minus_donor_logp':score(actual)-score(donor_memory),
                            'selection_rule':'hash order over sources retained by this selector, chosen without effect scores'}
                results[str(budget)]={'effects':effects,'intervention_positions':lengths,
                    'factual_positions':{s:len(flatten(chunks)) for s,chunks in factual.items()},
                    'coverage_tokens':{s:coverage(chunks,order) for s,chunks in factual.items()},'selection_changes':changes,
                    'answers':answers,'qa':{s:answer_quality(v['text'],item) for s,v in answers.items()},'length_controls':controls}
            row={'dataset':dataset,'item_id':item['id'],'split':item['split'],'stratum':item.get('hops',item.get('type')),
                'source_ids':order,'support_ids':item.get('support_ids'),'raw_answer':answer,'raw_qa':answer_quality(answer['text'],item),
                'raw_base':raw_base,'raw_positions':len(raw),'raw_effects':raw_effects,'budgets':results,
                'repeat_error':repeat_error,'raw_endpoint_error':endpoint_error}
            saved.append(row);atomic_text(trace,''.join(json.dumps(r,allow_nan=False)+'\n' for r in saved))
            print(dataset,item['split'],item['id'],'complete',len(saved),'seconds',round(time.monotonic()-started,1),flush=True)
    atomic_text(a.output/'completed.json',json.dumps({'complete':True,'n':len(saved),'elapsed_seconds':time.monotonic()-started})+'\n')


if __name__=='__main__':main()
