"""Quality/effect tradeoffs with stronger extraction and an explicit gold oracle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from ..attribution import sentences
from ..v2.hf import HFModel
from ..v2.memory import Contract,ExtractiveCodec,Source,digest
from ..v2.metrics import qa,effect_comparison
from ..v2.run import raw_memory


def answer_quality(text,item):
    scores=[qa(text,g) for g in [item['gold']]+item.get('answer_aliases',[])]
    return {'em':max(s['em'] for s in scores),'f1':max(s['f1'] for s in scores)}


def global_lexical(tokenizer,sources,question,budget):
    """A stronger conventional retrieval control, recomputed after deletion.

    Context-fitted IDF and global packing intentionally do not claim locality.
    """
    from collections import Counter
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
    for source,index,text,bag in ranked:
        ids=tokenizer.encode(f'\n[{source.sid}] {source.title}: {text}\n',add_special_tokens=False)
        if used+len(ids)<=budget:chosen.append((source.sid,index,ids));used+=len(ids)
    if not chosen and ranked:
        s,i,text,_=ranked[0];return tuple(tokenizer.encode(f'\n[{s.sid}] {s.title}: {text}\n',add_special_tokens=False)[:budget])
    chosen.sort(key=lambda x:(next(i for i,s in enumerate(sources) if s.sid==x[0]),x[1]))
    return tuple(token for _,_,ids in chosen for token in ids)


def oracle_sources(item,sources):
    """Annotation-informed ceiling, never an deployable label-free method."""
    selected=[]
    for source in sources:
        if source.sid not in item['support_ids']:continue
        original=next(s for s in item['sources'] if s['id']==source.sid)
        if item.get('supporting_facts'):
            indices=sorted({f['sentence_index'] for f in item['supporting_facts'] if f['source_id']==source.sid})
            text=' '.join(original['sentences'][k] for k in indices)
        else:text=source.text
        selected.append(Source(source.sid,source.title,text))
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model-key',choices=['qwen7','llama8','mistral7'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=96);args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Fresh evidence-study output required')
    items=[json.loads(s) for s in args.dataset.read_text().splitlines()][:args.limit]
    if not items or any(i['split']!='dev' for i in items):raise ValueError('Development only')
    args.output.mkdir(parents=True,exist_ok=True)
    registry=json.loads((args.root/'research/v2/models.json').read_text());paths=json.loads(args.paths.read_text());info=registry[args.model_key]
    model=HFModel(paths[args.model_key],info['revision'],model_id=info['model_id']);model.model.generation_config.repetition_penalty=1.
    manifest={'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'runtime':model.runtime(),
        'model_key':args.model_key,'budget':512,'limit':args.limit,'generation_repetition_penalty':1.,
        'source_sha256':{str(f.relative_to(args.root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for folder in ['v2','v4'] for f in (args.root/'src/context_attribution'/folder).glob('*.py')},
        'oracle_is_annotation_informed':True,'all_results':'exploratory development'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (args.output/'traces.jsonl').open('w') as handle:
        for item in items:
            sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
            codec=ExtractiveCodec(model.tokenizer,'lexical')
            policy=Contract(model.tokenizer,[s.sid for s in sources],mode='source',budget=512,codec_tag=codec.tag,question=item['question'])
            factual_lexical=policy.build(sources,codec)
            def memory(method,kept):
                if method=='raw':return raw_memory(model.tokenizer,kept)
                if method=='local_lexical':return policy.build(kept,codec,factual=factual_lexical).tokens
                if method=='global_lexical':return global_lexical(model.tokenizer,kept,item['question'],512)
                if method=='oracle_full':return raw_memory(model.tokenizer,oracle_sources(item,kept))
                raise ValueError(method)
            methods=['raw','local_lexical','global_lexical','oracle_full']
            target=None;raw_effects=None;raw_answer=None;raw_quality=None
            for method in methods:
                tokens=memory(method,sources);prefix=model.qa_ids(item['question'],tokens)
                answer=model.generate(prefix);quality=answer_quality(answer['text'],item)
                if method=='raw':target=answer['tokens'];raw_answer=answer;raw_quality=quality
                n=len(target);base=model.score(prefix,target);effects={};deleted={}
                for source in sources:
                    changed=memory(method,[s for s in sources if s.sid!=source.sid])
                    effects[source.sid]=(base-model.score(model.qa_ids(item['question'],changed),target))/n
                    deleted[source.sid]=changed
                if method=='raw':raw_effects=effects.copy()
                # Decision-level sanity check on a declared influential branch.
                top=max(raw_effects,key=lambda sid:abs(raw_effects[sid]))
                branch_answer=model.generate(model.qa_ids(item['question'],deleted[top]))
                non_support_mass=sum(abs(v) for sid,v in raw_effects.items() if sid not in item['support_ids'])
                total=sum(abs(v) for v in raw_effects.values())
                row={'item_id':item['id'],'method':method,'hops':item.get('hops'),'qa':quality,'raw_qa':raw_quality,
                    'answer':answer,'raw_answer':raw_answer,'memory_positions':len(tokens),'memory_text':model.tokenizer.decode(tokens),
                    'common_target_tokens':target,'effects':effects,'raw_effects':raw_effects,'comparison':effect_comparison(raw_effects,effects),
                    'support_ids':item['support_ids'],'support_granularity':item.get('support_granularity','sentence'),
                    'raw_effect_mass_on_non_support_fraction':non_support_mass/total if total else None,
                    'raw_max_influence_source':top,'branch_answer':branch_answer,
                    'oracle_scope':'oracle_full uses gold support annotations, no budget cap; diagnostic ceiling only' if method=='oracle_full' else None}
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
                print(item['id'],method,'F1',quality['f1'],flush=True)
    (args.output/'completed.json').write_text(json.dumps({'complete':True,'n':len(items),'conditions':len(items)*4}))


if __name__=='__main__':main()
