"""Four-reader, three-dataset comparison with shared targets and source ownership."""
import argparse
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import time

from .frontier_probe import flatten, selected_chunks
from .heldout_followup import digest
from .recomp_selector import MODEL_ID, REVISION, RecompSelector
from .selection_mechanism import pack_fixed, rank_chunks
from .v2.memory import Source
from .v2.run import raw_memory
from .v4.evidence import answer_quality

MODELS = ("qwen7", "llama8", "mistral7", "qwen14")
DATASETS = {"hotpot400": ("data/v2-hotpot/locked_test.jsonl", 400),
            "musique240": ("data/generalization/musique240.jsonl", 240),
            "2wiki200": ("data/generalization/2wiki200.jsonl", 200)}
SHARDS = 8
POLICIES = ("bm25_frozen", "bm25_fixed_rank", "bm25_refitted", "recomp_frozen", "recomp_repacked")


def read_data(project, name):
    relative, count = DATASETS[name]
    path = project / relative
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == count
    if name == "hotpot400":
        expected = "75598f99e35a5def46c440b42fc2c5168541dd751265d3d3e9f79964991cad5a"
    else:
        expected = json.loads((project / "data/generalization/manifest.json").read_text())["files"][name]
    assert digest(path.read_bytes()) == expected
    return rows, expected


def omissions(item):
    ids = [source["id"] for source in item["sources"]]
    single = [(sid,) for sid in ids]
    pairs = sorted(combinations(ids, 2), key=lambda pair: digest(
        ("generalization-joint-v1|" + item["id"] + "|" + ",".join(pair)).encode()))[:2]
    return single + pairs


def atomic_text(path, text):
    """Publish an entire checkpoint or leave its predecessor intact."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .v2.hf import HFModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--recomp-path", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = args.dataset is None
    assert smoke == (args.shard is None)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any(p.name != ".lock" for p in args.output.iterdir()):
        raise ValueError("Fresh output required; use --resume for matching checkpoints")
    datasets = DATASETS if smoke else [args.dataset]
    selected = []
    data_hashes = {}
    for name in datasets:
        rows, data_hashes[name] = read_data(args.project, name)
        if smoke:
            # Select before model scoring and include the longest serialized context.
            longest = max(rows, key=lambda row: sum(len(s["text"]) for s in row["sources"]))
            first = min(rows, key=lambda row: digest(("generalization-smoke-v1|" + row["id"]).encode()))
            rows = [first] if first["id"] == longest["id"] else [first, longest]
        else:
            rows = rows[args.shard::SHARDS]
        selected.extend((name, row) for row in rows)
    registry = json.loads((args.project / "research/v4/GENERALIZATION_MODELS.json").read_text())
    info = registry[args.model]
    paths = json.loads(args.paths.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    reader = HFModel(paths[args.model], info["revision"], model_id=info["model_id"])
    reader.model.float().requires_grad_(False)
    reader.model.generation_config.repetition_penalty = 1.
    selector = RecompSelector(args.recomp_path, reader.device)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "model": args.model, "dataset": args.dataset, "shard": args.shard,
        "runtime": reader.runtime(), "dataset_hashes": data_hashes,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "registry_sha256": digest((args.project / "research/v4/GENERALIZATION_MODELS.json").read_bytes()),
        "recomp": MODEL_ID, "recomp_revision": REVISION, "budget": 512,
        "policies": POLICIES, "expected_questions": len(selected),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    # Compare JSON-normalized values, including scientific source/data identity.
    receipt = json.loads(json.dumps(receipt))
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != receipt:
            raise ValueError("Resume manifest mismatch; refusing to mix experiments")
    else:
        atomic_text(manifest, json.dumps(receipt, indent=2) + "\n")
    trace = args.output / "traces.jsonl"
    saved = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    expected = [(name, item["id"]) for name, item in selected]
    if [(row["dataset"], row["item_id"]) for row in saved] != expected[:len(saved)] or len(saved) > len(expected):
        raise ValueError("Checkpoint question order mismatch")

    def direct_score(question, memory, target):
        prefix = reader.qa_ids(question, memory)
        ids = torch.tensor([prefix + list(target)], device=reader.device)
        length = ids.shape[-1]
        positions = torch.arange(length, device=reader.device)[None]
        mask = torch.ones((length, length), dtype=torch.bool, device=reader.device).tril()[None, None]
        with torch.inference_mode():
            logits = reader.model(input_ids=ids, position_ids=positions, attention_mask=mask,
                                  use_cache=False, logits_to_keep=len(target)+1).logits[0, :-1].double()
            assert logits.shape[0] == len(target)
            return float(logits.log_softmax(-1).gather(-1, ids[0, -len(target):, None]).mean())

    started = time.monotonic()
    with sdpa_kernel(SDPBackend.MATH):
        for name, item in selected[len(saved):]:
            question = item["question"]
            sources = [Source(s["id"], s["title"], s["text"]) for s in item["sources"]]
            order = [source.sid for source in sources]
            raw_memory_tokens = raw_memory(reader.tokenizer, sources)
            raw_answer = reader.generate(reader.qa_ids(question, raw_memory_tokens))
            target = raw_answer["tokens"]
            assert target
            raw_base = direct_score(question, raw_memory_tokens, target)
            repeat_error = abs(raw_base-direct_score(question, raw_memory_tokens, target))
            assert repeat_error <= 1e-7
            cache = {raw_memory_tokens: raw_base}

            def score(memory):
                tokens = tuple(memory)
                if tokens not in cache:
                    cache[tokens] = direct_score(question, tokens, target)
                return cache[tokens]

            bm25_rank = rank_chunks(reader.tokenizer, sources, question)
            recomp_rank = selector.rank(reader.tokenizer, sources, question)
            factual_bm25 = pack_fixed(bm25_rank, order, 512)
            factual_recomp = selector.pack(recomp_rank, sources, 512)
            assert flatten(factual_bm25) == flatten(selected_chunks(reader.tokenizer, sources, question, 512))
            factual = {"bm25": factual_bm25, "recomp": factual_recomp}
            answers = {key: reader.generate(reader.qa_ids(question, flatten(chunks))) for key, chunks in factual.items()}
            bases = {key: score(flatten(chunks)) for key, chunks in factual.items()}
            effects = {key: {} for key in ("raw",) + POLICIES}
            lengths = {key: {} for key in effects}
            plans = omissions(item)
            prefixes = {}
            for omit in plans:
                branch_key = ",".join(omit)
                kept = [source for source in sources if source.sid not in omit]
                bm25_filtered = [c for c in bm25_rank if c[0] not in omit]
                recomp_filtered = [c for c in recomp_rank if c[0] not in omit]
                memories = {
                    "raw": raw_memory(reader.tokenizer, kept),
                    "bm25_frozen": flatten([c for c in factual_bm25 if c[0] not in omit]),
                    "bm25_fixed_rank": flatten(pack_fixed(bm25_filtered, order, 512)),
                    "bm25_refitted": flatten(selected_chunks(reader.tokenizer, kept, question, 512)),
                    "recomp_frozen": flatten([c for c in factual_recomp if c[0] not in omit]),
                    "recomp_repacked": flatten(pack_fixed(recomp_filtered, order, 512)),
                }
                for policy, memory in memories.items():
                    base = raw_base if policy == "raw" else bases[policy.split("_")[0]]
                    effects[policy][branch_key] = base-score(memory)
                    lengths[policy][branch_key] = len(memory)
                    assert policy == "raw" or len(memory) <= 512
                prefixes[branch_key] = memories
            # Severe-source decision diagnostic has a declared selection mechanism.
            singleton = [source.sid for source in sources]
            top = max(singleton, key=lambda sid: abs(effects["raw"][sid]))
            branch_answers = {policy: reader.generate(reader.qa_ids(question, memory))
                              for policy, memory in prefixes[top].items()}
            sampled = min(singleton, key=lambda sid: digest((item["id"]+"|"+sid).encode()))
            endpoint_error = abs((raw_base-direct_score(question, prefixes[sampled]["raw"], target))-effects["raw"][sampled])
            assert endpoint_error <= 1e-7
            row = {
                "dataset": name, "item_id": item["id"], "hops": item.get("hops"), "type": item.get("type"),
                "raw_answer": raw_answer, "raw_qa": answer_quality(raw_answer["text"], item),
                "answers": answers, "qa": {k: answer_quality(v["text"], item) for k, v in answers.items()},
                "common_target_tokens": target, "raw_base": raw_base, "effects": effects,
                "factual_positions": {"raw": len(raw_memory_tokens), **{k: len(flatten(v)) for k, v in factual.items()}},
                "intervention_positions": lengths, "repeat_error": repeat_error, "raw_endpoint_error": endpoint_error,
                "top_raw_source": top, "branch_answers": branch_answers,
                "branch_qa": {k: answer_quality(v["text"], item) for k, v in branch_answers.items()},
            }
            saved.append(row)
            atomic_text(trace, "".join(json.dumps(r, allow_nan=False) + "\n" for r in saved))
            print(name, item["id"], "complete", round(time.monotonic()-started, 1), flush=True)
    atomic_text(args.output / "completed.json", json.dumps({"complete": True, "n": len(selected),
        "elapsed_seconds": time.monotonic()-started}) + "\n")


if __name__ == "__main__":
    main()
