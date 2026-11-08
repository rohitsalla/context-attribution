"""Development-only feasibility runner for structurally auditable compaction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from .memory import Contract,Source,ExtractiveCodec,digest
from .hf import HFModel,AbstractiveCodec
from .metrics import qa,effect_comparison


METHODS={'global':('global','abstractive'),'source':('source','abstractive'),
         'group4':('hash','abstractive'),'lead':('source','lead'),'lexical':('source','lexical')}


def raw_memory(tokenizer,sources):
    tokens=[]
    for s in sources:
        tokens.extend(tokenizer.encode(f'\n[{s.sid}] {s.title}: {s.text}\n',add_special_tokens=False))
    return tuple(tokens)


def start_cost(models):return {id(m):len(m.calls) for m in models}


def cost(models,start):
    values=[m.cost(start[id(m)]) for m in models]
    return {k:sum(v[k] for v in values) for k in ['calls','seconds','input_tokens','output_tokens']}


def raw_reference(item,answerer):
    sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
    tick=time.perf_counter();start=len(answerer.calls)
    memory=raw_memory(answerer.tokenizer,sources);prefix=answerer.qa_ids(item['question'],memory)
    generated=answerer.generate(prefix,stage='raw_answer')
    base=answerer.score(prefix,generated['tokens'],stage='raw_score')
    effects={}
    for source in sources:
        kept=[s for s in sources if s.sid!=source.sid]
        score=answerer.score(answerer.qa_ids(item['question'],raw_memory(answerer.tokenizer,kept)),
                             generated['tokens'],stage='raw_intervention_score')
        effects[source.sid]=base-score
    production_cost=answerer.cost(start);production_seconds=time.perf_counter()-tick
    diagnostic_start=len(answerer.calls)
    no_context=answerer.generate(answerer.qa_ids(item['question'],[]),stage='no_context_diagnostic')
    return {'item_id':item['id'],'method':'raw','budget':None,'answer':generated,
            'qa':qa(generated['text'],item['gold']),'memory_tokens':len(memory),
            'logprob':base,'effects':effects,'cost':production_cost,'seconds':production_seconds,
            'no_context':{'answer':no_context,'qa':qa(no_context['text'],item['gold']),
                          'diagnostic_cost':answerer.cost(diagnostic_start)}}


def evaluate(item,answerer,compressor,method,budget,raw,validation_deletions=2,seed=1729):
    sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
    mode,style=METHODS[method]
    codec=AbstractiveCodec(compressor,answerer.tokenizer) if style=='abstractive' else ExtractiveCodec(answerer.tokenizer,style)
    policy=Contract(answerer.tokenizer,[s.sid for s in sources],mode=mode,budget=budget,
                    codec_tag=codec.tag,question=item['question'],seed=seed)
    models=list({id(m):m for m in [answerer,compressor]}.values())
    stage_start=start_cost(models);tick=time.perf_counter()
    factual=policy.build(sources,codec)
    generated=answerer.generate(answerer.qa_ids(item['question'],factual.tokens),seed=seed,stage='compressed_answer')
    base=answerer.score(answerer.qa_ids(item['question'],factual.tokens),generated['tokens'],stage='compressed_score')
    production_seconds=time.perf_counter()-tick;initial_seconds=production_seconds
    initial_cost=cost(models,stage_start)
    memories={};effects={}
    for source in sources:
        tick=time.perf_counter();kept=[s for s in sources if s.sid!=source.sid]
        changed=policy.build(kept,codec,factual=factual);memories[source.sid]=changed
        value=answerer.score(answerer.qa_ids(item['question'],changed.tokens),generated['tokens'],stage='incremental_audit')
        effects[source.sid]=base-value;production_seconds+=time.perf_counter()-tick
    production_cost=cost(models,stage_start)
    # These diagnostic costs are explicitly outside production answer-plus-audit.
    diagnostic_start=start_cost(models)
    cold_factual=policy.build(sources,codec)
    if cold_factual.tokens!=factual.tokens:raise ValueError('Unchanged compaction replay is not reproducible')
    own_replay=answerer.score(answerer.qa_ids(item['question'],cold_factual.tokens),generated['tokens'],stage='unchanged_control')
    validation=[]
    selected=sorted(sources,key=lambda s:digest([item['id'],s.sid,seed]))[:validation_deletions]
    for source in selected:
        cold=policy.build([s for s in sources if s.sid!=source.sid],codec)
        cold_value=answerer.score(answerer.qa_ids(item['question'],cold.tokens),generated['tokens'],stage='uncached_validation')
        error=abs((base-cold_value)-effects[source.sid])
        validation.append({'source':source.sid,'identical_tokens':cold.tokens==memories[source.sid].tokens,
                           'effect_error':error,'cold_memory_sha256':digest(cold.tokens),
                           'cached_memory_sha256':digest(memories[source.sid].tokens)})
        if cold.tokens!=memories[source.sid].tokens or error>1e-4:
            raise ValueError('Incremental replay failed its own-pipeline validation')
    shared_target=generated['tokens']==raw['answer']['tokens']
    raw_target_base=base if shared_target else answerer.score(answerer.qa_ids(item['question'],factual.tokens),raw['answer']['tokens'],stage='common_target_diagnostic')
    common_effects={}
    for sid,memory in memories.items():
        common_effects[sid]=(effects[sid] if shared_target else raw_target_base-answerer.score(
            answerer.qa_ids(item['question'],memory.tokens),raw['answer']['tokens'],stage='common_target_diagnostic'))
    frozen_local=None
    if method=='global':
        from ..attribution import sentences
        import re
        body=answerer.tokenizer.decode(factual.chunks['g0']);parts=sentences(body)
        local_start=start_cost(models);tick=time.perf_counter();estimated={};mapped={}
        for source in sources:
            def mentions(part):return any(re.search(r'\b'+re.escape(source.sid)+r'\b',c) for c in re.findall(r'\[[^\]]+\]',part))
            mapped[source.sid]=any(mentions(p) for p in parts)
            edited=' '.join(p for p in parts if not mentions(p))
            ids=answerer.tokenizer.encode(edited,add_special_tokens=False)[:policy.quotas['g0']]
            changed=list(policy.headers['g0'])+ids
            estimated[source.sid]=base-answerer.score(answerer.qa_ids(item['question'],changed),generated['tokens'],stage='frozen_summary_baseline')
        frozen_local={'estimated_effects':estimated,'mapped_sources':mapped,
                      'comparison_to_same_pipeline_reference':effect_comparison(effects,estimated),
                      'original_plus_audit_seconds':initial_seconds+time.perf_counter()-tick,
                      'audit_only_cost':cost(models,local_start),'common_initial_cost':initial_cost}
    return {'item_id':item['id'],'method':method,'budget':budget,'answer':generated,'gold':item['gold'],
        'qa':qa(generated['text'],item['gold']),'memory_tokens':len(factual.tokens),
        'achieved_memory_ratio':len(factual.tokens)/max(raw['memory_tokens'],1),
        'memory_text':answerer.tokenizer.decode(factual.tokens),'quota_policy':policy.quotas,
        'factual_memory_tokens':list(factual.tokens),
        'factual_input_sha256':digest(answerer.qa_ids(item['question'],factual.tokens)),
        'intervention_receipts':{sid:{'memory_tokens':list(m.tokens),
            'input_sha256':digest(answerer.qa_ids(item['question'],m.tokens)),
            'dependencies':m.dependencies,'component_diagnostics':m.diagnostics} for sid,m in memories.items()},
        'component_dependencies':factual.dependencies,'component_diagnostics':factual.diagnostics,
        'initial_seconds':initial_seconds,'initial_cost':initial_cost,
        'production_seconds':production_seconds,'production_cost':production_cost,
        'own_pipeline_effects':effects,'unchanged_compaction_identical_tokens':True,
        'unchanged_logprob_shift':abs(own_replay-base),
        'sampled_uncached_validation':validation,'common_raw_target_tokens':raw['answer']['tokens'],
        'common_raw_target_effects':common_effects,'effect_distortion_vs_raw':effect_comparison(raw['effects'],common_effects),
        'diagnostic_cost':cost(models,diagnostic_start),'frozen_local_baseline':frozen_local}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--model-id',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--compressor',required=True);p.add_argument('--compressor-id',required=True);p.add_argument('--compressor-revision',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=8)
    p.add_argument('--budgets',nargs='+',type=int,default=[128,256,512])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--validation-deletions',type=int,default=2);args=p.parse_args()
    items=[json.loads(s) for s in args.dataset.read_text().splitlines()][:args.limit]
    if any(i['split']=='locked_test' for i in items):raise SystemExit('Locked test is disabled in the feasibility runner')
    args.output.mkdir(parents=True,exist_ok=True)
    fingerprint={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'source_sha256':{str(f.relative_to(Path(__file__).parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'attribution.py']}}
    mf=args.output/'manifest.json';trace=args.output/'traces.jsonl'
    if mf.exists() and json.loads(mf.read_text())!=fingerprint:raise ValueError('V2 resume fingerprint mismatch')
    rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
    completed={(r['item_id'],r['method'],r['budget']) for r in rows}
    raw_rows={r['item_id']:r for r in rows if r['method']=='raw'}
    expected={(i['id'],m,b) for i in items for m in args.methods for b in args.budgets}
    if expected<=completed:print('Feasibility shard already complete');return
    answerer=HFModel(args.model,args.revision,model_id=args.model_id)
    compressor=(answerer if args.model_id==args.compressor_id and args.revision==args.compressor_revision else
                HFModel(args.compressor,args.compressor_revision,model_id=args.compressor_id))
    runtime={'answerer':answerer.runtime(),'compressor':compressor.runtime()}
    with (args.output/'runtime_history.jsonl').open('a') as history:
        history.write(json.dumps(runtime)+'\n')
    mf.write_text(json.dumps(fingerprint,indent=2))
    with trace.open('a') as handle:
        def save(row):
            row['runtime']=runtime
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
            print(row['item_id'],row['method'],row['budget'],'F1',row['qa']['f1'],flush=True)
        for item in items:
            raw=raw_rows.get(item['id'])
            if raw is None:raw=raw_reference(item,answerer);save(raw)
            for budget in args.budgets:
                order=list(args.methods);random.Random(int(digest([item['id'],budget])[:8],16)).shuffle(order)
                for method in order:
                    if (item['id'],method,budget) in completed:continue
                    save(evaluate(item,answerer,compressor,method,budget,raw,args.validation_deletions))
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'conditions':len(items)*(1+len(args.methods)*len(args.budgets))},indent=2))


if __name__=='__main__':main()
