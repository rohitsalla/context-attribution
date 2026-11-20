"""Question-clustered review of the generalization matrix."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np

from context_attribution.generalization import DATASETS, MODELS, POLICIES, SHARDS, digest, omissions, read_data
from context_attribution.recomp_selector import MODEL_ID, REVISION
from context_attribution.v2.metrics import effect_comparison, normalize


def interval(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261120)
    samples = a[rng.integers(len(a), size=(5000, len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "question_bootstrap_95": np.quantile(samples, [.025, .975]).tolist()}


def review_cell(args, model, dataset, info, code_hash):
    items, data_hash = read_data(args.project, dataset)
    all_rows = {}
    missing = []
    for shard in range(SHARDS):
        folder = args.results / model / dataset / str(shard)
        if not (folder / "completed.json").is_file():
            missing.append(shard)
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        done = json.loads((folder / "completed.json").read_text())
        expected = items[shard::SHARDS]
        rows = [json.loads(line) for line in (folder / "traces.jsonl").read_text().splitlines()]
        assert done["complete"] and done["n"] == len(expected) == len(rows)
        assert manifest["model"] == model and manifest["dataset"] == dataset and manifest["shard"] == shard
        assert manifest["dataset_hashes"][dataset] == data_hash and manifest["code_sha256"] == code_hash
        assert manifest["runtime"]["revision"] == info["revision"] and manifest["runtime"]["dtype"] == "torch.float32"
        assert manifest["python_hash_seed"] == "0" and manifest["budget"] == 512
        assert manifest["recomp"] == MODEL_ID and manifest["recomp_revision"] == REVISION
        assert [r["item_id"] for r in rows] == [i["id"] for i in expected]
        for item, row in zip(expected, rows, strict=True):
            assert row["item_id"] not in all_rows and row["common_target_tokens"] == row["raw_answer"]["tokens"]
            assert row["repeat_error"] <= 1e-7 and row["raw_endpoint_error"] <= 1e-7
            keys = {",".join(omit) for omit in omissions(item)}
            assert set(row["effects"]) == {"raw", *POLICIES}
            assert all(set(v) == keys for v in row["effects"].values())
            assert all(row["factual_positions"][s] <= 512 for s in ("bm25", "recomp"))
            assert all(max(row["intervention_positions"][p].values()) <= 512 for p in POLICIES)
            all_rows[row["item_id"]] = row
    status = {"complete": not missing, "available_questions": len(all_rows), "missing_shards": missing}
    if missing:
        return status, None
    assert len(all_rows) == DATASETS[dataset][1]
    rows = [all_rows[qid] for qid in sorted(all_rows)]
    errors = {}
    status["methods"] = {}
    for policy in POLICIES:
        errors[policy] = {}
        policy_result = {}
        for group in ("singleton", "joint"):
            differences = []
            truth = []
            errors_by_question = []
            hits = []
            for row in rows:
                keys = [key for key in row["effects"]["raw"] if ("," in key) == (group == "joint")]
                ref = {k: row["effects"]["raw"][k] for k in keys}
                est = {k: row["effects"][policy][k] for k in keys}
                question_error = mean(abs(est[k]-ref[k]) for k in keys)
                errors_by_question.append(question_error)
                if group == "singleton":
                    errors[policy][row["item_id"]] = question_error
                differences.extend(abs(est[k]-ref[k]) for k in keys)
                truth.extend(abs(ref[k]) for k in keys)
                hit = effect_comparison(ref, est)["top1_hit"]
                if hit is not None:
                    hits.append(hit)
            policy_result[group] = {"question_mae": mean(errors_by_question),
                "pooled_nmae": sum(differences)/max(sum(truth), 1e-12),
                "top_effect_hit_rate": mean(hits) if hits else None}
        status["methods"][policy] = policy_result
    status["raw_qa_f1"] = mean(row["raw_qa"]["f1"] for row in rows)
    status["qa_f1"] = {s: mean(row["qa"][s]["f1"] for row in rows) for s in ("bm25", "recomp")}
    status["factual_positions"] = {s: mean(row["factual_positions"][s] for row in rows)
                                   for s in ("raw", "bm25", "recomp")}
    status["severe_source_decision_match_to_raw"] = {
        p: mean(normalize(r["branch_answers"][p]["text"]) == normalize(r["branch_answers"]["raw"]["text"]) for r in rows)
        for p in POLICIES}
    status["qa_on_severe_source_deletion"] = {
        p: mean(r["branch_qa"][p]["f1"] for r in rows) for p in ("raw",) + POLICIES}
    status["contrasts"] = {}
    for left, right in (("bm25_frozen", "bm25_refitted"), ("recomp_frozen", "recomp_repacked"),
                        ("bm25_fixed_rank", "bm25_frozen"), ("bm25_refitted", "bm25_fixed_rank")):
        status["contrasts"][left+"-minus-"+right] = interval(
            [errors[left][row["item_id"]]-errors[right][row["item_id"]] for row in rows])
    return status, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((args.project / "research/v4/GENERALIZATION_MODELS.json").read_text())
    code_hash = digest((args.project / "src/context_attribution/generalization.py").read_bytes())
    report = {"scope": "fixed-reader generalization study, paired question intervals",
              "cells": {}, "dataset_level_model_average": {}}
    error_cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            key = model + "/" + dataset
            status, errors = review_cell(args, model, dataset, registry[model], code_hash)
            report["cells"][key] = status
            error_cells[(model, dataset)] = errors
    for dataset in DATASETS:
        if any(error_cells[(model, dataset)] is None for model in MODELS):
            report["dataset_level_model_average"][dataset] = {"complete": False}
            continue
        ids = sorted(error_cells[(MODELS[0], dataset)]["bm25_frozen"])
        contrasts = {}
        for left, right in (("bm25_frozen", "bm25_refitted"), ("recomp_frozen", "recomp_repacked")):
            values = [mean(error_cells[(model, dataset)][left][qid] -
                           error_cells[(model, dataset)][right][qid] for model in MODELS) for qid in ids]
            contrasts[left+"-minus-"+right] = interval(values)
        report["dataset_level_model_average"][dataset] = {"complete": True, "contrasts": contrasts}
    complete = all(cell["complete"] for cell in report["cells"].values())
    passed = complete and all(
        contrast["question_bootstrap_95"][1] < 0
        for study in report["dataset_level_model_average"].values() for contrast in study["contrasts"].values())
    report["claim"] = ("replicates across the three datasets after averaging the four fixed readers"
                       if passed else "complete, generalization criterion unmet" if complete else "pending or incomplete")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
