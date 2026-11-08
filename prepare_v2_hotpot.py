"""Prepare revision-pinned HotpotQA splits; no model-based selection."""
import argparse
import hashlib
import json
from pathlib import Path

DATASET='hotpotqa/hotpot_qa'
REVISION='1908d6afbbead072334abe2965f91bd2709910ab'


def convert(row,split):
    titles=row['context']['title'];paragraphs=row['context']['sentences']
    if len(titles)!=len(paragraphs) or not titles:
        raise ValueError('Invalid source paragraph structure')
    sources=[{'id':f'd{i}','title':title,'text':' '.join(sentences).strip(),'sentences':sentences}
             for i,(title,sentences) in enumerate(zip(titles,paragraphs))]
    title_to_ids={title:[s['id'] for s in sources if s['title']==title] for title in titles}
    facts=[];issues=[]
    for title,index in zip(row['supporting_facts']['title'],row['supporting_facts']['sent_id']):
        candidates=[s for s in sources if s['title']==title and 0<=index<len(s['sentences'])]
        if not candidates:
            issues.append({'title':title,'sentence_index':index,'issue':'missing title or sentence in provided context'})
            continue
        facts.extend({'source_id':s['id'],'sentence_index':index} for s in candidates)
    return {'id':row['id'],'base_id':row['id'],'question':row['question'],'gold':row['answer'],
            'sources':sources,'type':row['type'],'level':row['level'],'split':split,
            'supporting_facts':facts,'support_ids':sorted(set(f['source_id'] for f in facts)),
            'duplicate_title_count':len(titles)-len(set(titles)),
            'ambiguous_support_titles':[title for title,ids in title_to_ids.items() if len(ids)>1],
            'annotation_issues':issues,
            'dataset':DATASET,'dataset_revision':REVISION,'human_reaudited':False}


def write(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-parquet',type=Path,required=True)
    p.add_argument('--validation-parquet',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/v2-hotpot'))
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite frozen v2 dataset')
    import pyarrow.parquet as pq
    def key(row):return hashlib.sha256(('context-v2-20261108|'+row['id']).encode()).hexdigest()
    train=sorted(pq.read_table(args.train_parquet).to_pylist(),key=key)
    validation=sorted(pq.read_table(args.validation_parquet).to_pylist(),key=key)
    # A fixed question-ID partition, chosen without inference or answer filtering.
    train_rows=[convert(r,'train') for r in train[:512]]
    dev_rows=[convert(r,'dev') for r in train[512:608]]
    test_rows=[convert(r,'locked_test') for r in validation[:400]]
    assert not ({r['id'] for r in train_rows}&{r['id'] for r in dev_rows})
    assert not ({r['id'] for r in train_rows+dev_rows}&{r['id'] for r in test_rows})
    args.output.mkdir(parents=True)
    for name,rows in [('train',train_rows),('dev',dev_rows),('locked_test',test_rows)]:write(args.output/f'{name}.jsonl',rows)
    # Gold support labels are retained for evaluation, never passed to codecs.
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.jsonl')}
    train_titles={s['title'] for r in train_rows for s in r['sources']}
    test_titles={s['title'] for r in test_rows for s in r['sources']}
    manifest={'dataset':DATASET,'revision':REVISION,'license':'CC-BY-SA-4.0 as reported by the HF dataset card',
        'source':'https://huggingface.co/datasets/hotpotqa/hotpot_qa','train_rows':512,'dev_rows':96,
        'locked_test_rows':400,'files':files,'test_model_evaluations_authorized':False,
        'selection':'SHA256 ID order; train uses the first pinned train shard, test uses official validation',
        'source_parquet_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train_parquet,args.validation_parquet]},
        'train_test_shared_document_titles':len(train_titles&test_titles),
        'annotation_issue_rows':{name:sum(bool(r['annotation_issues']) for r in rows)
                                 for name,rows in [('train',train_rows),('dev',dev_rows),('locked_test',test_rows)]},
        'notes':['Question IDs are disjoint. Retrieval-corpus document overlap is reported, not hidden.',
                 'The original paragraph list is preserved, including duplicate titles and variable source counts.',
                 'Out-of-range source annotations are flagged, not silently repaired or used as gold source labels.',
                 'The locked test split must not be run until development gates are met.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','source_parquet_sha256']},indent=2))


if __name__=='__main__':main()
