"""Exploratory analysis of where frozen deletion improves fidelity."""
import argparse
import json
from pathlib import Path
from statistics import mean

from review_selection_mechanism import interval

FROZEN = "global_512_frozen"
RECOMPUTED = "global_512_recomputed"


def load_pairs(paths, expected_n):
    pairs = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["method"] not in (FROZEN, RECOMPUTED):
                continue
            pair = pairs.setdefault(row["item_id"], {})
            assert row["method"] not in pair
            pair[row["method"]] = row
    assert len(pairs) == expected_n
    for pair in pairs.values():
        frozen, recomputed = pair[FROZEN], pair[RECOMPUTED]
        assert frozen["raw_effects"] == recomputed["raw_effects"]
        assert frozen["factual_memory_sha256"] == recomputed["factual_memory_sha256"]
        assert frozen["source_positions"] == recomputed["source_positions"]
    return dict(sorted(pairs.items()))


def gains(pair):
    frozen, recomputed = pair[FROZEN], pair[RECOMPUTED]
    result = {"represented": 0.0, "unrepresented": 0.0}
    raw = frozen["raw_effects"]
    for source, reference in raw.items():
        group = "represented" if frozen["source_positions"][source] else "unrepresented"
        improvement = abs(recomputed["effects"][source] - reference) - abs(frozen["effects"][source] - reference)
        result[group] += improvement / len(raw)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = {}
    for model in ("qwen7", "llama8"):
        heldout = load_pairs(sorted((args.root / "heldout-selection" / model).glob("*/traces.jsonl")), 400)
        bm25 = load_pairs([args.root / "frontier" / f"hotpot_{model}" / "traces.jsonl"], 96)
        recomp = load_pairs([args.root / "recomp-selection" / f"hotpot_{model}" / "traces.jsonl"], 96)
        assert bm25.keys() == recomp.keys()
        assert all(bm25[q][FROZEN]["common_target_tokens"] == recomp[q][FROZEN]["common_target_tokens"] for q in bm25)
        contributions = [gains(pair) for pair in heldout.values()]
        group_means = {name: mean(item[name] for item in contributions) for name in contributions[0]}
        report[model] = {
            "heldout_bm25_contribution_to_mean_gain": group_means,
            "heldout_bm25_represented_share_of_gain": group_means["represented"] / sum(group_means.values()),
            "dev_bm25_minus_recomp_gain": interval([sum(gains(bm25[q]).values()) - sum(gains(recomp[q]).values()) for q in bm25]),
            "dev_max_raw_effect_difference": max(abs(value - recomp[q][FROZEN]["raw_effects"][source])
                for q, pair in bm25.items() for source, value in pair[FROZEN]["raw_effects"].items()),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
