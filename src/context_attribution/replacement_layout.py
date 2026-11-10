"""Bridge frozen deletion to fixed-ranking repacking with explicit masking."""

import argparse
import hashlib
import json
from pathlib import Path

from .frontier_probe import flatten
from .selection_mechanism import pack_fixed, rank_chunks
from .v2.memory import Source

STUDIES = ("hotpot_qwen7", "hotpot_llama8", "musique_qwen7", "musique_llama8")
ARCHIVE_TOLERANCE = 1e-6
SHAPE_TOLERANCE = 1e-4
MASK_TOLERANCE = 1e-7


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cohort(items, benchmark):
    def priority(item):
        return sha256(("replacement-layout-v1:" + item["id"]).encode())

    ranked = sorted(items, key=priority)
    if benchmark == "hotpot":
        return ranked[:24]
    return [item for hop in (2, 3, 4)
            for item in [r for r in ranked if r["hops"] == hop][:8]]


def bridge_chunks(factual, repacked, omit):
    # Include the token tuple in identity to handle a truncated fallback chunk.
    kept = [chunk for chunk in factual if chunk[0] != omit]
    visible = set(kept)
    common = [chunk for chunk in repacked if chunk in visible]
    flags = [chunk in visible for chunk in repacked for _ in chunk[2]]
    if common != [chunk for chunk in kept if chunk in set(common)]:
        raise ValueError("Shared chunks changed order")
    return flatten(kept), flatten(common), flags


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .v2.hf import HFModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Fresh output required")
    benchmark, key = args.study.split("_")
    reference = args.root / "results/selection-mechanism" / args.study
    frontier = args.root / "results/frontier" / args.study
    manifest = json.loads((reference / "manifest.json").read_text())
    dataset = args.root / ("data/v2-hotpot/dev.jsonl" if benchmark == "hotpot"
                           else "data/v4-musique/frontier32.jsonl")
    assert sha256(dataset.read_bytes()) == manifest["dataset_sha256"]
    assert sha256((frontier / "traces.jsonl").read_bytes()) == manifest["reference_trace_sha256"]
    items = read_rows(dataset)
    refs = {row["item_id"]: row for row in read_rows(reference / "traces.jsonl")}
    done = json.loads((reference / "completed.json").read_text())
    assert done["complete"] and done["n"] == len(items) == len(refs)
    assert all(item["split"] == "dev" for item in items)
    assert set(refs) == {item["id"] for item in items}
    donor_pool = sorted(items, key=lambda item: item["id"])
    items = cohort(items, benchmark)
    assert len(items) == 24
    if args.smoke:
        items = items[:2]

    info = json.loads((args.root / "research/v2/models.json").read_text())[key]
    paths = json.loads(args.paths.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    model = HFModel(paths[key], info["revision"], model_id=info["model_id"])
    model.model.float().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "study": args.study, "runtime": model.runtime(),
        "item_ids": [item["id"] for item in items],
        "source_sha256": sha256(Path(__file__).read_bytes()),
        "dataset_sha256": manifest["dataset_sha256"],
        "reference_sha256": sha256((reference / "traces.jsonl").read_bytes()),
        "archive_tolerance": ARCHIVE_TOLERANCE,
        "shape_tolerance": SHAPE_TOLERANCE,
        "mask_tolerance": MASK_TOLERANCE,
    }
    (args.output / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    def score(prefix, target, visible=None, compact_ids=False):
        ids = torch.tensor([list(prefix) + list(target)], device=model.device)
        length = ids.shape[-1]
        positions = torch.arange(length, device=model.device)
        mask = torch.ones((length, length), dtype=torch.bool, device=model.device).tril()
        if visible is not None:
            live = torch.tensor(visible + [True] * len(target), device=model.device)
            mask &= live[None, :]
            if compact_ids:
                positions = live.long().cumsum(0) - 1
        with torch.inference_mode():
            logits = model.model(input_ids=ids, position_ids=positions[None],
                attention_mask=mask[None, None], use_cache=False).logits[0, len(prefix)-1:-1].double()
            return float(logits.log_softmax(-1).gather(-1, ids[0, len(prefix):, None]).mean())

    with sdpa_kernel(SDPBackend.MATH), (args.output / "traces.jsonl").open("w") as handle:
        for index, item in enumerate(items, 1):
            ref = refs[item["id"]]
            target = ref["common_target_tokens"]
            sources = [Source(s["id"], s["title"], s["text"]) for s in item["sources"]]
            order = [source.sid for source in sources]
            ranked = rank_chunks(model.tokenizer, sources, item["question"])
            factual = pack_fixed(ranked, order, 512)
            memory = flatten(factual)
            assert sha256(json.dumps(memory).encode()) == ref["factual_memory_sha256"]
            left, right = model.qa_sides(item["question"])
            assert left and right and target
            assert left + list(memory) + right == model.qa_ids(item["question"], memory)
            base = score(left + list(memory) + right, target)
            if abs(base - ref["factual_target_logp"]) > ARCHIVE_TOLERANCE:
                raise ValueError("Archived factual score drift")

            titles = {source.title.casefold().strip() for source in sources}
            donor_tokens = []
            donor_ids = []
            for donor in donor_pool:
                if donor["id"] == item["id"]:
                    continue
                for source in donor["sources"]:
                    if source["title"].casefold().strip() not in titles:
                        donor_tokens.extend(model.tokenizer.encode(source["text"], add_special_tokens=False))
                donor_ids.append(donor["id"])
                if len(donor_tokens) >= 512:
                    break
            assert len(donor_tokens) >= 512
            branches = {}
            mask_error = None
            for sid in order:
                repacked = pack_fixed(ranked, order, 512, sid)
                short, shared, visible_memory = bridge_chunks(factual, repacked, sid)
                long = flatten(repacked)
                assert tuple(t for t, live in zip(long, visible_memory, strict=True) if live) == shared
                frozen_effect = ref["frozen_effects"][sid]
                repack_effect = ref["effects"][sid]
                compact = frozen_effect if short == shared else base - score(left + list(shared) + right, target)
                shape_error = 0.0
                if all(visible_memory):
                    dense_compact = expanded = activated = donor_effect = compact
                else:
                    prefix = left + list(long) + right
                    visible = [True] * len(left) + visible_memory + [True] * len(right)
                    dense_compact = base - score(prefix, target, visible, compact_ids=True)
                    expanded = base - score(prefix, target, visible)
                    activated = base - score(prefix, target)
                    donor_stream = iter(donor_tokens)
                    donor_prefix = [token if live else next(donor_stream)
                                    for token, live in zip(prefix, visible, strict=True)]
                    donor_effect = base - score(donor_prefix, target)
                    shape_error = abs(dense_compact - compact)
                    if mask_error is None:
                        substitute = [token if live else left[0]
                                      for token, live in zip(prefix, visible, strict=True)]
                        mask_error = abs(expanded - (base - score(substitute, target, visible)))
                        if mask_error > MASK_TOLERANCE:
                            raise ValueError("Masked token identity changed the score")
                archive_error = abs(activated - repack_effect)
                if archive_error > ARCHIVE_TOLERANCE:
                    raise ValueError("Archived repack score drift")
                branches[sid] = {
                    "raw": ref["raw_effects"][sid], "frozen": frozen_effect,
                    "common_compact": compact, "common_dense_compact_ids": dense_compact,
                    "common_dense_expanded_ids": expanded, "replacement_visible": activated,
                    "matched_donor_visible": donor_effect,
                    "frozen_positions": len(short), "shared_positions": len(shared),
                    "repacked_positions": len(long), "shape_error": shape_error,
                    "archive_error": archive_error,
                }
            row = {"item_id": item["id"], "branches": branches, "mask_error": mask_error,
                   "donor_item_ids": donor_ids}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(index, item["id"], "complete", flush=True)
    (args.output / "completed.json").write_text(json.dumps({"complete": True, "n": len(items)}) + "\n")


if __name__ == "__main__":
    main()
