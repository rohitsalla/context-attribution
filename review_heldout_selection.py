"""Evaluate the prespecified frozen-versus-recomputed Hotpot claim."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from context_attribution.frontier_probe import HELDOUT_SHA256, HELDOUT_SHARDS
from context_attribution.v2.metrics import effect_comparison
from review_selection_mechanism import interval

MODELS = ("qwen7", "llama8")
FROZEN = "global_512_frozen"
RECOMPUTED = "global_512_recomputed"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def review_model(root, model, questions, source_hash, revision):
    incomplete = []
    records = {}
    receipts = []
    for shard in range(HELDOUT_SHARDS):
        folder = root / model / str(shard)
        completion = folder / "completed.json"
        if not completion.is_file():
            incomplete.append(shard)
            continue
        expected = questions[shard::HELDOUT_SHARDS]
        manifest = json.loads((folder / "manifest.json").read_text())
        done = json.loads(completion.read_text())
        trace = folder / "traces.jsonl"
        rows = read_rows(trace)
        assert manifest["dataset_sha256"] == HELDOUT_SHA256
        assert manifest["source_sha256"] == source_hash
        assert manifest["heldout_shard"] == shard and manifest["budgets"] == [512]
        assert manifest["runtime"]["revision"] == revision
        assert manifest["python_hash_seed"] == "0"
        assert done["complete"] and done["n"] == len(expected)
        assert len(rows) == done["conditions"] == 3 * len(expected)
        groups = {}
        for row in rows:
            group = groups.setdefault(row["item_id"], {})
            assert row["method"] not in group
            group[row["method"]] = row
        assert set(groups) == {item["id"] for item in expected}
        for item in expected:
            qid = item["id"]
            assert qid not in records
            group = groups[qid]
            assert set(group) == {"raw", FROZEN, RECOMPUTED}
            raw, frozen, recomputed = (group[name] for name in ("raw", FROZEN, RECOMPUTED))
            assert raw["raw_identity_endpoint_error"] <= 1e-7
            assert frozen["factual_memory_sha256"] == recomputed["factual_memory_sha256"]
            assert frozen["answer"] == recomputed["answer"]
            assert frozen["qa"] == recomputed["qa"]
            source_ids = {s["id"] for s in item["sources"]}
            assert set(raw["raw_effects"]) == source_ids
            for row in (raw, frozen, recomputed):
                assert row["same_shape_repeat_error"] <= 1e-7
                assert row["common_target_tokens"] == raw["answer"]["tokens"]
                assert row["raw_effects"] == raw["raw_effects"]
                assert set(row["effects"]) == source_ids
            for row in (frozen, recomputed):
                assert row["memory_positions"] <= 512
                assert max(row["intervention_positions"].values()) <= 512
            records[qid] = (raw, frozen, recomputed)
        receipts.append({"shard": shard, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
    result = {"complete": not incomplete, "validated_questions": len(records),
              "missing_shards": incomplete, "trace_receipts": receipts}
    if incomplete:
        return result
    assert len(records) == len(questions) == 400
    ordered = [records[qid] for qid in sorted(records)]
    raw_l1 = sum(abs(effect) for raw, _, _ in ordered for effect in raw["raw_effects"].values())
    assert raw_l1 > 0
    result["methods"] = {}
    errors = {}
    for index, name in ((1, FROZEN), (2, RECOMPUTED)):
        errors[name] = []
        total_error = 0.
        ranks = []
        for group in ordered:
            raw, method = group[0], group[index]
            differences = [abs(method["effects"][sid]-effect)
                           for sid, effect in raw["raw_effects"].items()]
            errors[name].append(mean(differences))
            total_error += sum(differences)
            hit = effect_comparison(raw["raw_effects"], method["effects"])["top1_hit"]
            if hit is not None:
                ranks.append(hit)
        result["methods"][name] = {
            "question_mae": mean(errors[name]), "pooled_nmae": total_error / raw_l1,
            "top_source_hit_rate": mean(ranks) if ranks else None,
            "mean_factual_positions": mean(g[index]["memory_positions"] for g in ordered),
            "mean_intervention_positions": mean(mean(g[index]["intervention_positions"].values()) for g in ordered),
        }
    result["paired_frozen_minus_recomputed_mae"] = interval(
        [a-b for a,b in zip(errors[FROZEN], errors[RECOMPUTED], strict=True)]
    )
    result["raw_qa_f1"] = mean(g[0]["qa"]["f1"] for g in ordered)
    result["shared_compressed_qa_f1"] = mean(g[1]["qa"]["f1"] for g in ordered)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.project / "data/v2-hotpot/locked_test.jsonl"
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == HELDOUT_SHA256
    questions = read_rows(dataset)
    source = args.project / "src/context_attribution/frontier_probe.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = json.loads((args.project / "research/v2/models.json").read_text())
    studies = {model: review_model(args.root, model, questions, source_hash, registry[model]["revision"])
               for model in MODELS}
    if not all(study["complete"] for study in studies.values()):
        verdict = "pending or incomplete"
    elif all(study["paired_frozen_minus_recomputed_mae"]["question_bootstrap_95"][1] < 0
             for study in studies.values()):
        verdict = "replicated on held-out Hotpot with both models"
    else:
        verdict = "cross-model confirmation criterion unmet"
    result = {"scope": "held-out Hotpot, shared 400 questions across models",
              "claim": verdict, "studies": studies}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
