"""Freeze development/test cohorts before any new reader scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from prepare_generalization import MU_REVISION, WIKI_REVISION, fingerprint


def sha(data):
    return hashlib.sha256(data).hexdigest()


def priority(row):
    return sha(('stability-20261124-v1|' + row['id']).encode())


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    out = a.project / 'data/stability'
    if out.exists(): raise ValueError('Fresh cohort directory required')
    seen = set(); excluded_files = {}
    for file in sorted((a.project / 'data').rglob('*.jsonl')):
        excluded_files[str(file.relative_to(a.project))] = sha(file.read_bytes())
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if isinstance(row, dict) and 'question' in row: seen.add(fingerprint(row['question']))
    mu_path = Path(hf_hub_download('dgslibisey/MuSiQue', 'musique_ans_v1.0_dev.jsonl',
                   revision=MU_REVISION, repo_type='dataset', token=False))
    wiki_path = Path(hf_hub_download('framolfese/2WikiMultihopQA', 'data/validation-00000-of-00001.parquet',
                     revision=WIKI_REVISION, repo_type='dataset', token=False))
    mu = [json.loads(s) for s in mu_path.read_text().splitlines()]
    wiki = pq.read_table(wiki_path).to_pylist()
    selected = {'musique': [], '2wiki': []}; counts = {}; hashes = {}; donor_maps = {}
    for name, raw in [('musique', mu), ('2wiki', wiki)]:
        strata = [2, 3, 4] if name == 'musique' else sorted({r['type'] for r in raw})
        assert len(strata) == (3 if name == 'musique' else 4)
        for stratum in strata:
            candidates = [r for r in raw if (r['answerable'] and int(r['id'][0]) == stratum)] if name == 'musique' else [r for r in raw if r['type'] == stratum]
            candidates.sort(key=priority); cursor = 0
            for split, per_stratum in [('dev', 28 if name == 'musique' else 20), ('test', 80 if name == 'musique' else 60)]:
                added = 0
                while added < per_stratum:
                    row = candidates[cursor]; cursor += 1; q = fingerprint(row['question'])
                    if q in seen: continue
                    seen.add(q)
                    if name == 'musique':
                        sources = [{'id': f'd{i}', 'title': s['title'], 'text': s['paragraph_text']} for i,s in enumerate(row['paragraphs'])]
                        supports = [f'd{i}' for i,s in enumerate(row['paragraphs']) if s.get('is_supporting')]
                        extra = {'hops': stratum, 'answer_aliases': row['answer_aliases']}
                    else:
                        sources = [{'id': f'd{i}', 'title': title, 'text': ' '.join(sentences)} for i,(title,sentences) in enumerate(zip(row['context']['title'], row['context']['sentences'], strict=True))]
                        sf = row.get('supporting_facts', {}); titles = sf.get('title', []) if isinstance(sf,dict) else [s[0] for s in sf]
                        supports = [s['id'] for s in sources if s['title'] in titles]
                        extra = {'type': stratum, 'answer_aliases': []}
                    selected[name].append({'id': row['id'], 'question': row['question'], 'gold': row['answer'], 'sources': sources,
                         'support_ids': supports or None, 'split': split, 'benchmark': name, **extra})
                    added += 1
        selected[name].sort(key=lambda r: (r['split'], priority(r)))
        assert len({r['id'] for r in selected[name]}) == len(selected[name])
        counts[name] = dict(Counter(r['split'] for r in selected[name]))
        assert counts[name] == {'dev': 84 if name == 'musique' else 80, 'test': 240}
        donor_maps[name] = {}
        for row in selected[name]:
            pool = [r for r in selected[name] if r['split'] == row['split'] and r['id'] != row['id']]
            donor = min(pool, key=lambda r: sha(('donor-v1|' + row['id'] + '|' + r['id']).encode()))
            donor_maps[name][row['id']] = donor['id']
    out.mkdir(parents=True)
    for name, rows in selected.items():
        path = out / (name + '.jsonl'); path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)); hashes[name] = sha(path.read_bytes())
    manifest = {'cohort_rule': 'stability-20261124-v1 SHA256 ID order within strata, normalized-question exclusion of every existing project data JSONL',
        'counts': counts, 'files': hashes, 'excluded_files': excluded_files, 'donor_map': donor_maps,
        'raw_hashes': {'musique': sha(mu_path.read_bytes()), '2wiki': sha(wiki_path.read_bytes())},
        'revisions': {'musique': MU_REVISION, '2wiki': WIKI_REVISION},
        'prior_question_overlap': 0, 'dev_test_question_overlap': 0,
        'provenance': 'revision-pinned third-party mirrors of official dev/validation data; byte equivalence and pretraining/document contamination unverified',
        'licenses': {'musique': 'CC-BY-4.0 original release', '2wiki': 'Apache-2.0 original release'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'counts':counts,'files':hashes},indent=2))


if __name__ == '__main__': main()
