"""Freeze new dataset cohorts using identifiers and declared strata only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

MU_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
WIKI_REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"


def fingerprint(text):
    return " ".join(text.casefold().split())


def priority(row):
    return hashlib.sha256(("context-generalization-v1|" + row["id"]).encode()).hexdigest()


def main():
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    output = args.project / "data/generalization"
    if output.exists():
        raise ValueError("Fresh dataset output required")
    excluded = set()
    for relative in ("data/v2-hotpot/train.jsonl", "data/v2-hotpot/dev.jsonl",
                     "data/v2-hotpot/locked_test.jsonl", "data/v4-musique/dev.jsonl"):
        excluded.update(fingerprint(json.loads(line)["question"])
                        for line in (args.project / relative).read_text().splitlines())
    mu_path = Path(hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl",
                                 revision=MU_REVISION, repo_type="dataset", token=False))
    mu = [json.loads(line) for line in mu_path.read_text().splitlines()]
    selected_mu = []
    seen = set(excluded)
    for hops in (2, 3, 4):
        candidates = [r for r in mu if r["answerable"] and int(r["id"][0]) == hops]
        candidates.sort(key=priority)
        for row in candidates:
            question = fingerprint(row["question"])
            if question in seen:
                continue
            seen.add(question)
            selected_mu.append({
                "id": row["id"], "question": row["question"], "gold": row["answer"],
                "answer_aliases": row["answer_aliases"], "hops": hops, "split": "generalization",
                "sources": [{"id": f"d{i}", "title": s["title"], "text": s["paragraph_text"]}
                            for i, s in enumerate(row["paragraphs"])],
                "dataset": "dgslibisey/MuSiQue", "revision": MU_REVISION,
            })
            if sum(r["hops"] == hops for r in selected_mu) == 80:
                break
    assert Counter(r["hops"] for r in selected_mu) == {2: 80, 3: 80, 4: 80}
    wiki_path = Path(hf_hub_download("framolfese/2WikiMultihopQA", "data/validation-00000-of-00001.parquet",
                                   revision=WIKI_REVISION, repo_type="dataset", token=False))
    wiki = parquet.read_table(wiki_path).to_pylist()
    selected_wiki = []
    types = sorted({r["type"] for r in wiki})
    assert len(types) == 4
    for kind in types:
        for row in sorted([r for r in wiki if r["type"] == kind], key=priority):
            question = fingerprint(row["question"])
            if question in seen:
                continue
            seen.add(question)
            context = row["context"]
            titles, texts = context["title"], context["sentences"]
            assert len(titles) == len(texts)
            selected_wiki.append({
                "id": row["id"], "question": row["question"], "gold": row["answer"],
                "answer_aliases": [], "type": kind, "split": "generalization",
                "sources": [{"id": f"d{i}", "title": title, "text": " ".join(sentences)}
                            for i, (title, sentences) in enumerate(zip(titles, texts, strict=True))],
                "dataset": "framolfese/2WikiMultihopQA", "revision": WIKI_REVISION,
            })
            if sum(r["type"] == kind for r in selected_wiki) == 50:
                break
    assert len(selected_wiki) == 200
    output.mkdir(parents=True)
    hashes = {}
    for name, rows in (("musique240", selected_mu), ("2wiki200", selected_wiki)):
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "selection": "fixed SHA256 ID order within hop or question-type strata, normalized-question deduplication",
        "counts": {"musique240": 240, "2wiki200": 200}, "files": hashes,
        "revisions": {"musique": MU_REVISION, "2wiki": WIKI_REVISION},
        "raw_hashes": {"musique": hashlib.sha256(mu_path.read_bytes()).hexdigest(),
                       "2wiki": hashlib.sha256(wiki_path.read_bytes()).hexdigest()},
        "previous_project_question_overlap": 0, "cross_new_benchmark_question_overlap": 0,
        "official_release_byte_equivalence_verified": False,
        "provenance": "Pinned third-party mirrors of official development/validation data",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
