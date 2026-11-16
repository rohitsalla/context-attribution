"""Paired question-level review of the archived-target follow-up."""
import argparse
import json
from pathlib import Path
from statistics import mean

from context_attribution.frontier_probe import HELDOUT_SHA256, HELDOUT_SHARDS
from context_attribution.heldout_followup import TOLERANCE, digest, load_archive
from context_attribution.recomp_selector import MODEL_ID, REVISION
from review_selection_mechanism import interval

METHODS = ("bm25_frozen", "bm25_fixed_rank", "bm25_refitted", "recomp_frozen", "recomp_repacked")


def evaluate(args, model, questions, revision, code_hash, archive_receipts):
    incomplete = [s for s in range(HELDOUT_SHARDS)
                  if not (args.results / model / str(s) / "completed.json").is_file()]
    if incomplete:
        return {"complete": False, "missing_shards": incomplete}

    errors = {method: {} for method in METHODS}
    totals = dict.fromkeys(METHODS, 0.)
    represented = {method: {} for method in METHODS if method.startswith("bm25")}
    max_drift = 0.
    raw_mass = 0.
    for shard in range(HELDOUT_SHARDS):
        items = questions[shard::HELDOUT_SHARDS]
        folder = args.results / model / str(shard)
        archive, archive_hash = load_archive(args.archive / model / str(shard), items,
                                             revision, HELDOUT_SHA256, shard)
        manifest = json.loads((folder / "manifest.json").read_text())
        done = json.loads((folder / "completed.json").read_text())
        rows = [json.loads(line) for line in (folder / "traces.jsonl").read_text().splitlines()]
        assert done["complete"] and done["n"] == len(rows) == len(items)
        assert [row["item_id"] for row in rows] == [item["id"] for item in items]
        assert manifest["model"] == model and manifest["shard"] == shard
        assert manifest["runtime"]["revision"] == revision and manifest["code_sha256"] == code_hash
        assert manifest["data_sha256"] == HELDOUT_SHA256
        assert manifest["archive_sha256"] == archive_hash == archive_receipts[shard]
        assert manifest["recomp"] == MODEL_ID and manifest["recomp_revision"] == REVISION
        assert manifest["archive_tolerance"] == TOLERANCE and manifest["budget"] == 512
        for row in rows:
            qid = row["item_id"]
            raw, frozen, refitted = (archive[qid][key] for key in
                                     ("raw", "global_512_frozen", "global_512_recomputed"))
            reference = raw["raw_effects"]
            assert row["raw_target"] == raw["common_target_tokens"]
            assert 0 <= row["archive_drift"] <= TOLERANCE
            max_drift = max(max_drift, row["archive_drift"])
            raw_mass += sum(map(abs, reference.values()))
            estimates = {"bm25_frozen": frozen["effects"], "bm25_refitted": refitted["effects"], **row["effects"]}
            assert set(estimates) == set(METHODS)
            assert 0 < row["recomp_factual_positions"] <= 512
            for method in METHODS:
                assert set(estimates[method]) == set(reference)
                assert qid not in errors[method]
                difference = {sid: abs(estimates[method][sid] - value) for sid, value in reference.items()}
                errors[method][qid] = mean(difference.values())
                totals[method] += sum(difference.values())
                if method.startswith("bm25"):
                    represented[method][qid] = sum(
                        value for sid, value in difference.items() if frozen["source_positions"][sid]) / len(reference)
    assert all(len(values) == 400 for values in errors.values()) and raw_mass > 0
    ids = sorted(errors["bm25_frozen"])

    def contrasts(metrics):
        refill = [metrics["bm25_fixed_rank"][q] - metrics["bm25_frozen"][q] for q in ids]
        rerank = [metrics["bm25_refitted"][q] - metrics["bm25_fixed_rank"][q] for q in ids]
        return {"refill": interval(refill), "rerank": interval(rerank),
                "refill_minus_rerank": interval([a - b for a, b in zip(refill, rerank, strict=True)])}

    bm25_gain = [errors["bm25_refitted"][q] - errors["bm25_frozen"][q] for q in ids]
    recomp_gain = [errors["recomp_repacked"][q] - errors["recomp_frozen"][q] for q in ids]
    return {
        "complete": True, "n": len(ids), "max_archive_drift": max_drift,
        "mae": {m: mean(errors[m].values()) for m in METHODS},
        "nmae": {m: totals[m] / raw_mass for m in METHODS},
        "bm25_mae_increase": contrasts(errors),
        "bm25_represented_source_contribution": contrasts(represented),
        "recomp_frozen_gain": interval(recomp_gain),
        "bm25_gain_minus_recomp_gain": interval([a - b for a, b in zip(bm25_gain, recomp_gain, strict=True)]),
    }


def lower(study, key):
    return study[key]["question_bootstrap_95"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = (args.project / "data/v2-hotpot/locked_test.jsonl").read_bytes()
    assert digest(data) == HELDOUT_SHA256
    questions = [json.loads(line) for line in data.splitlines()]
    archive_review = json.loads((args.archive / "review.json").read_text())
    models = json.loads((args.project / "research/v2/models.json").read_text())
    code_hash = digest((args.project / "src/context_attribution/heldout_followup.py").read_bytes())
    studies = {m: evaluate(args, m, questions, models[m]["revision"], code_hash,
                           {r["shard"]: r["trace_sha256"] for r in archive_review["studies"][m]["trace_receipts"]})
               for m in ("qwen7", "llama8")}
    report = {"scope": "secondary follow-up on shared 400 Hotpot questions", "studies": studies}
    if all(s["complete"] for s in studies.values()):
        bm25 = [s["bm25_mae_increase"] for s in studies.values()]
        if all(lower(s, "refill") > 0 and lower(s, "refill_minus_rerank") > 0 for s in bm25):
            report["bm25_direction"] = "refilling contributes more than refitting rankings on both readers"
        elif all(lower(s, "rerank") > 0 and s["refill_minus_rerank"]["question_bootstrap_95"][1] < 0 for s in bm25):
            report["bm25_direction"] = "refitting rankings contributes more than refilling on both readers"
        else:
            report["bm25_direction"] = "dominant contribution unresolved across readers"
        report["recomp_direction"] = ("frozen deletion improves RECOMP MAE on both readers"
                                       if all(lower(s, "recomp_frozen_gain") > 0 for s in studies.values())
                                       else "cross-reader RECOMP criterion unmet")
    else:
        report.update(bm25_direction="pending or incomplete", recomp_direction="pending or incomplete")
    args.results.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    (args.results / "review.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
