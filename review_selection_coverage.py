"""CPU-only ownership audit of completed BF16 global-extraction studies."""
import argparse
import hashlib
import json
from pathlib import Path
from context_attribution.frontier_probe import selected_chunks,flatten
from context_attribution.v2.memory import Source
from context_attribution.v4.evidence import global_lexical


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--paths',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths=json.loads(args.paths.read_text());registry=json.loads((args.root/'research/v2/models.json').read_text())
    result={'scope':'post-hoc CPU ownership audit of previous BF16 traces; not new FP32 evidence or a selection-causality estimate',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'zero_threshold':.001,'studies':{}}
    for key in ['qwen7','llama8','mistral7']:
        tokenizer=AutoTokenizer.from_pretrained(paths[key],revision=registry[key]['revision'],local_files_only=True)
        for benchmark,data in [('hotpot','data/v2-hotpot/dev.jsonl'),('musique','data/v4-musique/dev.jsonl')]:
            folder=args.root/'results/v4'/f'evidence_{benchmark}_{key}'
            assert json.loads((folder/'completed.json').read_text())['complete']
            trace=folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
            rows=[r for r in rows if r['method']=='global_lexical'];assert len(rows)==96
            items={r['id']:r for r in map(json.loads,(args.root/data).read_text().splitlines())}
            total=absent=nonzero=0;raw_absent=raw_total=compressed_absent=compressed_total=0.;per_question=[]
            for row in rows:
                item=items[row['item_id']];sources=[Source(s['id'],s['title'],s['text']) for s in item['sources']]
                chunks=selected_chunks(tokenizer,sources,item['question'],512);tokens=flatten(chunks)
                assert tokens==global_lexical(tokenizer,sources,item['question'],512)
                assert tokenizer.decode(tokens)==row['memory_text'] and len(tokens)==row['memory_positions']
                owners={sid for sid,_,t in chunks if t};excluded=set(row['effects'])-owners
                assert set(row['effects'])=={s.sid for s in sources}
                m=sum(abs(row['effects'][k]) for k in excluded);r=sum(abs(row['raw_effects'][k]) for k in excluded)
                all_m=sum(abs(v) for v in row['effects'].values());all_r=sum(abs(v) for v in row['raw_effects'].values())
                total+=len(row['effects']);absent+=len(excluded);nonzero+=sum(abs(row['effects'][k])>.001 for k in excluded)
                compressed_absent+=m;raw_absent+=r;compressed_total+=all_m;raw_total+=all_r
                per_question.append({'item_id':item['id'],'sources_without_selected_tokens':sorted(excluded),
                    'raw_l1_on_excluded':r,'raw_l1_total':all_r,'compressed_l1_on_excluded':m,'compressed_l1_total':all_m})
            result['studies'][f'{benchmark}_{key}']={'n':96,'all_factual_memories_reconstructed_exactly':True,
                'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                'n_source_deletions':total,'n_sources_without_selected_tokens':absent,
                'n_excluded_with_absolute_compressed_effect_above_threshold':nonzero,
                'fraction_raw_l1_on_excluded':raw_absent/max(raw_total,1e-12),
                'fraction_compressed_l1_on_excluded':compressed_absent/max(compressed_total,1e-12),
                'per_question':per_question}
    result['interpretation']='Sources with no selected factual tokens can influence a recomputed selector. Nonzero BF16 effects alone do not isolate selection changes from numerical sensitivity. Frozen-selection influence for those sources is zero by construction. Their raw L1 share is a simple lower bound on frozen raw-transfer NMAE, not a novel theorem. Do not assume freezing improves fidelity.'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_question'} for k,v in result['studies'].items()},indent=2))


if __name__=='__main__':main()
