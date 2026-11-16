"""Score fixed-rank BM25 and two RECOMP deletion policies on archived targets."""
import argparse
import hashlib
import json
from pathlib import Path

from .frontier_probe import HELDOUT_SHA256, HELDOUT_SHARDS, flatten, selected_chunks
from .recomp_selector import MODEL_ID, REVISION, RecompSelector
from .selection_mechanism import pack_fixed, rank_chunks
from .v2.memory import Source
from .v2.run import raw_memory

TOLERANCE = 1e-4
SELECTION_CAP = 512


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_archive(folder, items, model_revision, dataset_hash, shard):
    trace = (folder / "traces.jsonl").read_bytes()
    manifest = json.loads((folder / "manifest.json").read_text())
    done = json.loads((folder / "completed.json").read_text())
    assert done["complete"] and done["n"] == len(items)
    assert manifest["dataset_sha256"] == dataset_hash
    assert manifest["heldout_shard"] == shard and manifest["budgets"] == [512]
    assert manifest["runtime"]["revision"] == model_revision
    assert manifest["python_hash_seed"] == "0"
    index = {}
    for line in trace.splitlines():
        row = json.loads(line)
        methods = index.setdefault(row["item_id"], {})
        assert row["method"] not in methods
        methods[row["method"]] = row
    assert set(index) == {item["id"] for item in items}
    assert all(set(group) == {"raw", "global_512_frozen", "global_512_recomputed"}
               for group in index.values())
    return index, digest(trace)


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .v2.hf import HFModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--recomp-path", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen7", "llama8"], required=True)
    parser.add_argument("--shard", type=int, choices=range(HELDOUT_SHARDS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    smoke = args.shard is None
    dataset = args.project / "data/v2-hotpot" / ("dev.jsonl" if smoke else "locked_test.jsonl")
    dataset_bytes = dataset.read_bytes()
    dataset_hash = digest(dataset_bytes)
    items = [json.loads(line) for line in dataset_bytes.splitlines()]
    if smoke:
        items = items[:2]
        archive_folder = args.archive / "heldout-selection-smoke" / args.model
    else:
        assert dataset_hash == HELDOUT_SHA256 and len(items) == 400
        assert all(item["split"] == "locked_test" for item in items)
        items = items[args.shard::HELDOUT_SHARDS]
        archive_folder = args.archive / "heldout-selection" / args.model / str(args.shard)

    model_info = json.loads((args.project / "research/v2/models.json").read_text())[args.model]
    archive, archive_hash = load_archive(archive_folder, items, model_info["revision"], dataset_hash, args.shard)
    if not smoke:
        verified = json.loads((args.archive / "heldout-selection/review.json").read_text())
        receipts = verified["studies"][args.model]["trace_receipts"]
        assert {r["shard"]: r["trace_sha256"] for r in receipts}[args.shard] == archive_hash
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Fresh output required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    paths = json.loads(args.paths.read_text())
    reader = HFModel(paths[args.model], model_info["revision"], model_id=model_info["model_id"])
    reader.model.float().requires_grad_(False)
    reader.model.generation_config.repetition_penalty = 1.
    selector = RecompSelector(args.recomp_path, reader.device)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": args.model, "runtime": reader.runtime(), "shard": args.shard,
        "data_sha256": dataset_hash, "archive_sha256": archive_hash,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "recomp": MODEL_ID, "recomp_revision": REVISION, "budget": SELECTION_CAP,
        "archive_tolerance": TOLERANCE, "target": "archived raw factual greedy answer",
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def score(question, memory, target):
        prefix = reader.qa_ids(question, memory)
        ids = torch.tensor([prefix + list(target)], device=reader.device)
        length = ids.shape[-1]
        positions = torch.arange(length, device=reader.device)[None]
        mask = torch.ones((length, length), dtype=torch.bool, device=reader.device).tril()[None, None]
        with torch.inference_mode():
            logits = reader.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                  use_cache=False).logits[0, len(prefix)-1:-1].double()
            return float(logits.log_softmax(-1).gather(-1, ids[0, len(prefix):, None]).mean())

    with sdpa_kernel(SDPBackend.MATH), (args.output / "traces.jsonl").open("w") as handle:
        for item in items:
            qid, question = item["id"], item["question"]
            raw, frozen, recomputed = (archive[qid][name] for name in
                                       ("raw", "global_512_frozen", "global_512_recomputed"))
            target = raw["common_target_tokens"]
            assert target and target == raw["answer"]["tokens"]
            assert all(row["common_target_tokens"] == target and row["raw_effects"] == raw["raw_effects"]
                       for row in (frozen, recomputed))
            sources = [Source(s["id"], s["title"], s["text"]) for s in item["sources"]]
            order = [s.sid for s in sources]
            assert set(order) == set(raw["raw_effects"])
            bm25_rank = rank_chunks(reader.tokenizer, sources, question)
            bm25_tokens = flatten(pack_fixed(bm25_rank, order, SELECTION_CAP))
            assert digest(json.dumps(bm25_tokens).encode()) == frozen["factual_memory_sha256"] == recomputed["factual_memory_sha256"]
            bm25_base = score(question, bm25_tokens, target)
            raw_base = score(question, raw_memory(reader.tokenizer, sources), target)
            sampled = min(order, key=lambda sid: digest(f"{qid}:{sid}".encode()))
            survivors = [s for s in sources if s.sid != sampled]
            bm25_old = selected_chunks(reader.tokenizer, survivors, question, SELECTION_CAP)
            sampled_frozen = flatten(pack_fixed(bm25_rank, order, SELECTION_CAP), sampled)
            drift = max(
                abs(bm25_base - frozen["factual_target_logp"]),
                abs(raw_base - raw["raw_factual_target_logp"]),
                abs(raw_base - score(question, raw_memory(reader.tokenizer, survivors), target)
                    - raw["raw_effects"][sampled]),
                abs(bm25_base - score(question, bm25_old, target) - recomputed["effects"][sampled]),
                abs(bm25_base - score(question, sampled_frozen, target) - frozen["effects"][sampled]),
            )
            if drift > TOLERANCE:
                raise RuntimeError(f"Archived score drift at {qid}: {drift}")
            recomp_rank = selector.rank(reader.tokenizer, sources, question)
            recomp_factual = selector.pack(recomp_rank, sources, SELECTION_CAP)
            recomp_tokens = flatten(recomp_factual)
            assert len(recomp_tokens) <= SELECTION_CAP
            recomp_base = score(question, recomp_tokens, target)
            if abs(recomp_base - score(question, recomp_tokens, target)) > 1e-7:
                raise ValueError("RECOMP factual score is not repeatable")
            effects = {name: {} for name in ("bm25_fixed_rank", "recomp_frozen", "recomp_repacked")}
            lengths = {name: {} for name in effects}
            for sid in order:
                branches = {
                    "bm25_fixed_rank": (bm25_base, flatten(pack_fixed(bm25_rank, order, SELECTION_CAP, sid))),
                    "recomp_frozen": (recomp_base, flatten(recomp_factual, sid)),
                    "recomp_repacked": (recomp_base, flatten(selector.pack(recomp_rank, sources, SELECTION_CAP, sid))),
                }
                score_cache = {bm25_tokens: bm25_base, recomp_tokens: recomp_base}
                for name, (base, tokens) in branches.items():
                    assert len(tokens) <= SELECTION_CAP
                    if tokens not in score_cache:
                        score_cache[tokens] = score(question, tokens, target)
                    effects[name][sid] = base - score_cache[tokens]
                    lengths[name][sid] = len(tokens)
            row = {"item_id": qid, "effects": effects, "archive_drift": drift,
                   "raw_target": target, "recomp_factual_positions": len(recomp_tokens),
                   "intervention_positions": lengths}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(qid, "complete", flush=True)
    (args.output / "completed.json").write_text(json.dumps({"complete": True, "n": len(items)}) + "\n")


if __name__ == "__main__":
    main()
