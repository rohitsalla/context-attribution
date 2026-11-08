"""Exploratory question-conditioning and entity-alias diagnostics, development only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics as st
from .v2.hf import HFModel,AbstractiveCodec
from .v2.memory import Contract,ExtractiveCodec,Source,digest
from .v2.metrics import qa,effect_comparison
from .v2.run import raw_reference,raw_memory


def alias_item(item):
    """Mechanical stress transformation, not a new natural-text benchmark.

    Gold may define an entity renaming here, never compressor routing/selection.
    Exact strings only: partial names and coreferences are not resolved.
    """
    result=copy.deepcopy(item);names={s['title'] for s in item['sources'] if s['title']}
    gold=item['gold']
    if gold and gold[0].isupper() and gold.lower() not in ['yes','no','noanswer']:names.add(gold)
    mapping={name:'Zerova'+digest([item['id'],name])[:8] for name in names}
    pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(names,key=lambda n:(-len(n),n)))+r')(?!\w)')
    def replace(text):return pattern.sub(lambda match:mapping[match.group()],text)
    result['question']=replace(item['question']);result['gold']=replace(item['gold'])
    for source in result['sources']:
        source['title']=replace(source['title']);source['text']=replace(source['text'])
    result['id']=item['id']+'-aliases'
    result['transformation']={'kind':'exact title/entity aliases; incomplete coreference coverage','mapping':mapping,
                              'origin':item['id'],'human_audited':False}
    return result


def summarize(output):
    path=output/'traces.jsonl';values=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    groups={}
    for row in values:groups.setdefault((row['variant'],row['method']),[]).append(row)
    report={'scope':'adaptive eight-question mechanism diagnostic','complete':(output/'completed.json').exists(),'conditions':[]}
    for (variant,method),rows in groups.items():
        result={'variant':variant,'method':method,'n':len(rows),'f1':st.mean(r['qa']['f1'] for r in rows),
                'em':st.mean(r['qa']['em'] for r in rows)}
        if method!='raw':
            error=sum(abs(r['common_target_effects'][k]-v) for r in rows for k,v in r['raw_effects'].items())
            scale=sum(abs(v) for r in rows for v in r['raw_effects'].values())
            both=[r for r in rows if r['qa']['em']==1 and r['raw_qa']['em']==1]
            result.update({'pooled_nmae':error/max(scale,1e-8),'mean_positions':st.mean(r['memory_positions'] for r in rows),
                'both_correct_n':len(both),'both_correct_rank_misses':sum(r['comparison']['top1_hit'] is False for r in both),
                'answer_generation_caps':sum(r['answer']['cap_reached'] for r in rows)})
        report['conditions'].append(result)
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=8);p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
    if args.summarize_only:summarize(args.output);return
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh probe output required')
    args.output.mkdir(parents=True,exist_ok=True)
    items=[json.loads(s) for s in (args.root/'data/v2-hotpot/dev.jsonl').read_text().splitlines()][:args.limit]
    if not items or any(i['split']!='dev' for i in items):raise ValueError('Development only')
    registry=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text())
    answerer=HFModel(paths['llama8'],registry['llama8']['revision'],model_id=registry['llama8']['model_id'])
    compressor=HFModel(paths['qwen15'],registry['qwen15']['revision'],model_id=registry['qwen15']['model_id'])
    for model in [answerer,compressor]:model.model.generation_config.repetition_penalty=1.0
    manifest={'type':'exploratory mechanism probe, no general efficacy claim','limit':args.limit,'budget':512,
              'runtime':{'answerer':answerer.runtime(),'compressor':compressor.runtime()},
              'generation_policy':{'do_sample':False,'repetition_penalty':1.0,'max_answer_tokens':64},
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256((args.root/'data/v2-hotpot/dev.jsonl').read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (args.output/'traces.jsonl').open('w') as handle:
        def save(row):
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
        for original in items:
            for variant,item in [('original',original),('aliases',alias_item(original))]:
                sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
                raw=raw_reference(item,answerer);raw['variant']=variant
                most_influential=max(raw['effects'],key=lambda sid:abs(raw['effects'][sid]))
                independent=sorted([s.sid for s in sources if s.sid!=most_influential],key=lambda sid:digest([item['id'],sid,'branch-probe']))[0]
                selected=[most_influential,independent];raw_branches={}
                for sid in selected:
                    kept=[s for s in sources if s.sid!=sid]
                    raw_branches[sid]=answerer.generate(answerer.qa_ids(item['question'],raw_memory(answerer.tokenizer,kept)))
                raw['branch_answers']=raw_branches;save(raw)
                probes={}
                for name,model in [('answerer',answerer),('compressor',compressor)]:
                    answer=model.generate(model.chat_ids('Answer with a short answer only.\nQuestion: '+item['question']))
                    probes[name]={'answer':answer,'qa':qa(answer['text'],item['gold']),
                                  'note':'Unconstrained knowledge probe; prompt differs from evidence-only QA. Not a knowledge-origin certificate.'}
                for method in ['global_aware','global_blind','source_aware','lexical']:
                    question='' if method=='global_blind' else item['question']
                    mode='global' if method.startswith('global') else 'source'
                    codec=ExtractiveCodec(answerer.tokenizer,'lexical') if method=='lexical' else AbstractiveCodec(compressor,answerer.tokenizer)
                    policy=Contract(answerer.tokenizer,[s.sid for s in sources],mode=mode,budget=512,
                                    codec_tag=codec.tag+':'+method,question=question)
                    factual=policy.build(sources,codec);target=raw['answer']['tokens']
                    answer=answerer.generate(answerer.qa_ids(item['question'],factual.tokens))
                    base=answerer.score(answerer.qa_ids(item['question'],factual.tokens),target)
                    effects={};deleted={};branches={}
                    for source in sources:
                        changed=policy.build([s for s in sources if s.sid!=source.sid],codec,factual=factual)
                        effects[source.sid]=base-answerer.score(answerer.qa_ids(item['question'],changed.tokens),target)
                        deleted[source.sid]=answerer.tokenizer.decode(changed.tokens)
                        if source.sid in selected:branches[source.sid]=answerer.generate(answerer.qa_ids(item['question'],changed.tokens))
                    row={'item_id':item['id'],'origin':original['id'],'variant':variant,'method':method,
                         'answer':answer,'qa':qa(answer['text'],item['gold']),'raw_qa':raw['qa'],
                         'memory_text':answerer.tokenizer.decode(factual.tokens),'memory_positions':len(factual.tokens),
                         'raw_effects':raw['effects'],'common_target_effects':effects,'common_target_tokens':target,
                         'comparison':effect_comparison(raw['effects'],effects),'deleted_memory_text':deleted,
                         'knowledge_probes':probes,'transformation':item.get('transformation')}
                    row.update({'branch_answers':branches,'raw_branch_answers':raw_branches,
                                'branch_selection':{'raw_max_magnitude':most_influential,'hash_selected_control':independent},
                                'branch_scope':'Diagnostic generations, not charged as production attribution cost. Original gold is not a new gold label for evidence-deleted inputs.'})
                    save(row);print(original['id'],variant,method,'F1',row['qa']['f1'],flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'original_questions':len(items),'conditions':len(items)*10}))
    summarize(args.output)


if __name__=='__main__':main()
