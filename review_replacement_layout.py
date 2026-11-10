"""Aggregate the replacement/layout control and apply its direction rule."""

import argparse
import json
from pathlib import Path
from statistics import mean

from context_attribution.replacement_layout import SHAPE_TOLERANCE, STUDIES, read_rows
from review_selection_mechanism import interval


def study_result(folder):
    trace = folder / "traces.jsonl"
    rows = read_rows(trace) if trace.exists() else []
    completion = folder / "completed.json"
    if not completion.exists():
        return {"complete": False, "available_questions": len(rows)}
    done = json.loads(completion.read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert done["complete"] and done["n"] == len(rows) == 24
    assert [row["item_id"] for row in rows] == manifest["item_ids"]
    methods = ("frozen", "common_compact", "common_dense_compact_ids",
               "common_dense_expanded_ids", "replacement_visible", "matched_donor_visible")
    mae = {name: [mean(abs(b[name] - b["raw"]) for b in row["branches"].values())
                  for row in rows] for name in methods}
    contrasts = {
        "eviction": ("common_compact", "frozen"),
        "shape_residual": ("common_dense_compact_ids", "common_compact"),
        "position_expansion": ("common_dense_expanded_ids", "common_dense_compact_ids"),
        "replacement_activation": ("replacement_visible", "common_dense_expanded_ids"),
        "matched_length_evidence_identity": ("replacement_visible", "matched_donor_visible"),
        "donor_activation": ("matched_donor_visible", "common_dense_expanded_ids"),
        "full_repacking": ("replacement_visible", "frozen"),
    }
    differences = {name: [a - b for a, b in zip(mae[left], mae[right], strict=True)]
                   for name, (left, right) in contrasts.items()}
    differences["activation_minus_position"] = [
        activation - position for activation, position in
        zip(differences["replacement_activation"], differences["position_expansion"], strict=True)
    ]
    branches = [branch for row in rows for branch in row["branches"].values()]
    shape_error = max(b["shape_error"] for b in branches)
    checked_questions = sum(row["mask_error"] is not None for row in rows)
    return {
        "complete": True, "n": len(rows), "n_deletions": len(branches),
        "n_replacement_deletions": sum(b["repacked_positions"] > b["shared_positions"] for b in branches),
        "n_eviction_deletions": sum(b["frozen_positions"] > b["shared_positions"] for b in branches),
        "n_mask_checked_questions": checked_questions,
        "max_archive_error": max(b["archive_error"] for b in branches),
        "max_shape_error": shape_error,
        "validation_passed": shape_error <= SHAPE_TOLERANCE and checked_questions > 0,
        "mean_question_mae": {name: mean(values) for name, values in mae.items()},
        "mean_positions": {name: mean(b[name] for b in branches)
                           for name in ("frozen_positions", "shared_positions", "repacked_positions")},
        "paired_mae_differences": {name: interval(values) for name, values in differences.items()},
    }


def choose_direction(studies):
    if not all(study["complete"] and study["validation_passed"] for study in studies.values()):
        return "incomplete or validation failed"

    def bounds(study, contrast):
        return study["paired_mae_differences"][contrast]["question_bootstrap_95"]

    def both_benchmarks(names):
        return {name.split("_")[0] for name in names} == {"hotpot", "musique"}

    activation = [name for name, s in studies.items()
                  if bounds(s, "replacement_activation")[0] > 0
                  and bounds(s, "activation_minus_position")[0] > 0]
    position = [name for name, s in studies.items()
                if bounds(s, "position_expansion")[0] > 0
                and bounds(s, "activation_minus_position")[1] < 0]
    if both_benchmarks(activation) and not position:
        identity = [name for name in activation
                    if bounds(studies[name], "matched_length_evidence_identity")[0] > 0]
        opposite = any(bounds(s, "matched_length_evidence_identity")[1] < 0 for s in studies.values())
        if both_benchmarks(identity) and not opposite:
            return "replacement evidence identity"
        return "attended replacement slots, identity contribution unresolved or mixed"
    if both_benchmarks(position) and not activation:
        return "post-deletion positional layout"
    return "mixed or uncertain, retain the selection-contract finding"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    studies = {name: study_result(args.root / name) for name in STUDIES}
    report = {"scope": "exploratory development, unadjusted paired question intervals",
              "paper_direction": choose_direction(studies), "studies": studies}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
