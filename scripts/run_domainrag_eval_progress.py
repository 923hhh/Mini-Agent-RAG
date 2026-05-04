"""Run DomainRAG retrieval with visible progress bars.

This script is a lightweight wrapper for local experiments against a checked-out
`DomainRAG-main` workspace. It focuses on the retrieval stage and adds an
optional cross-encoder rerank pass with explicit tqdm progress bars.

Example:
    python scripts/run_domainrag_eval_progress.py ^
        --domainrag-root "E:\\南京航空航天大学\\aaa大创\\智能体案例\\DomainRAG-main" ^
        --corpus-json-root "E:\\南京航空航天大学\\aaa大创\\智能体案例\\DomainRAG-main\\corpus\\rdzs\\json_output" ^
        --retriever bm25 ^
        --reranker-path ".\\data\\models\\bge-reranker-v2-m3" ^
        --max-samples 1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


DEFAULT_TASK_FILES = {
    "basic": "BCM/labeled_data/extractive_qa/basic_qa.jsonl",
    "conversation": "BCM/labeled_data/conversation_qa/conversation_qa.jsonl",
    "multidoc": "BCM/labeled_data/multi-doc_qa/multidoc_qa.jsonl",
    "time": "BCM/labeled_data/time-sensitive_qa/time_sensitive.jsonl",
    "structure": "BCM/labeled_data/structured_qa/structured_qa_twopositive.jsonl",
}


@dataclass(frozen=True)
class PassageRecord:
    doc_key: str
    title: str
    url: str
    date: str
    contents: str
    psg_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domainrag-root",
        type=Path,
        required=True,
        help="Local path of DomainRAG-main.",
    )
    parser.add_argument(
        "--corpus-json-root",
        type=Path,
        default=None,
        help="Directory that contains DomainRAG corpus json files. "
        "Default: <domainrag-root>/corpus/rdzs/json_output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/domainrag_progress_runs"),
        help="Directory for merged samples and retrieval outputs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASK_FILES.keys()),
        choices=list(DEFAULT_TASK_FILES.keys()),
        help="Which official DomainRAG tasks to include.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Upper bound of evaluation samples to run.",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="hybrid",
        help="Retriever type.",
    )
    parser.add_argument(
        "--dense-model-path",
        type=str,
        default="BAAI/bge-base-zh-v1.5",
        help="Embedding model for dense retrieval.",
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        default="",
        help="Optional CrossEncoder path/name. Leave empty to skip rerank.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Retrieved candidates before rerank.",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=50,
        help="Dense candidate depth before fusion when retriever=hybrid.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=50,
        help="BM25 candidate depth before fusion when retriever=hybrid.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant used in hybrid fusion.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5,
        help="Saved top-k after optional rerank.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Dense embedding batch size.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=16,
        help="Cross-encoder rerank batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dense/rerank device, for example cpu or cuda.",
    )
    return parser.parse_args()


def resolve_corpus_root(domainrag_root: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    return domainrag_root / "corpus" / "rdzs" / "json_output"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_query(task: str, sample: dict[str, Any]) -> str:
    if task == "time":
        year = str(sample.get("date", "")).strip()
        prefix = f"{year}年 " if year else ""
        return normalize_space(prefix + str(sample.get("question", "")))
    if task == "conversation":
        history = [
            str(item.get("question", "")).strip()
            for item in sample.get("history_qa", [])
            if str(item.get("question", "")).strip()
        ]
        history.append(str(sample.get("question", "")).strip())
        return normalize_space(" ".join(history))
    return normalize_space(str(sample.get("question", "")))


def load_eval_samples(domainrag_root: Path, tasks: list[str], max_samples: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for task in tasks:
        dataset_path = domainrag_root / DEFAULT_TASK_FILES[task]
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        rows = load_jsonl(dataset_path)
        for index, row in enumerate(rows):
            merged.append(
                {
                    "task": task,
                    "sample_index": index,
                    "question": row.get("question", ""),
                    "query": build_query(task, row),
                    "raw": row,
                }
            )
    return merged[: max(max_samples, 0)]


def load_corpus_passages(corpus_json_root: Path) -> list[PassageRecord]:
    if not corpus_json_root.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_json_root}. "
            "Download and decompress DomainRAG corpus first."
        )

    json_paths = sorted(corpus_json_root.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No corpus json files found under: {corpus_json_root}")

    passages: list[PassageRecord] = []
    for path in tqdm(json_paths, desc="Load corpus docs", unit="doc"):
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(data.get("id", ""))
        url = str(data.get("url", "")).strip()
        doc_key = "\t".join([doc_id, url])
        title = normalize_space(data.get("title", ""))
        date = normalize_space(data.get("date", ""))
        for psg_id, passage in enumerate(data.get("passages", [])):
            passages.append(
                PassageRecord(
                    doc_key=doc_key,
                    title=title,
                    url=url,
                    date=date,
                    contents=normalize_space(str(passage).replace("\u3000", " ")),
                    psg_id=psg_id,
                )
            )
    return passages


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize_for_bm25(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return ["__empty__"]
    return TOKEN_PATTERN.findall(text) or ["__empty__"]


class BM25Retriever:
    def __init__(self, passages: list[PassageRecord]):
        from rank_bm25 import BM25Okapi

        tokenized = []
        for passage in tqdm(passages, desc="Tokenize passages", unit="psg"):
            tokenized.append(tokenize_for_bm25(f"{passage.title} {passage.contents}"))
        self._bm25 = BM25Okapi(tokenized)
        self._passages = passages

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize_for_bm25(query))
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class DenseRetriever:
    def __init__(self, passages: list[PassageRecord], model_path: str, batch_size: int, device: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)
        self._passages = passages
        texts = [format_passage_text(item) for item in passages]
        self._embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_embedding = self._model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]
        scores = np.dot(self._embeddings, query_embedding)
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


def fuse_hybrid_candidates(
    dense_candidates: list[tuple[int, float]],
    bm25_candidates: list[tuple[int, float]],
    top_k: int,
    rrf_k: int,
) -> list[tuple[int, float]]:
    fused_scores: dict[int, float] = {}

    for rank, (passage_idx, _) in enumerate(dense_candidates, start=1):
        fused_scores[passage_idx] = fused_scores.get(passage_idx, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (passage_idx, _) in enumerate(bm25_candidates, start=1):
        fused_scores[passage_idx] = fused_scores.get(passage_idx, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(passage_idx, float(score)) for passage_idx, score in ranked[:top_k]]


def format_passage_text(passage: PassageRecord) -> str:
    return f"{passage.title}。{passage.contents}"


def rerank_candidates(
    queries: list[str],
    candidate_lists: list[list[tuple[int, float]]],
    passages: list[PassageRecord],
    reranker_path: str,
    batch_size: int,
    device: str,
) -> list[list[tuple[int, float]]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(reranker_path, device=device)
    reranked_results: list[list[tuple[int, float]]] = []
    for query, candidates in tqdm(
        list(zip(queries, candidate_lists)),
        desc="Rerank queries",
        unit="query",
    ):
        if not candidates:
            reranked_results.append([])
            continue
        pairs = [(query, format_passage_text(passages[idx])) for idx, _ in candidates]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(
            [
                (candidate_idx, float(score))
                for (candidate_idx, _), score in zip(candidates, scores, strict=False)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        reranked_results.append(ranked)
    return reranked_results


def materialize_result(
    sample: dict[str, Any],
    ranked_candidates: list[tuple[int, float]],
    passages: list[PassageRecord],
    final_top_k: int,
    retrieval_type: str,
    reranked: bool,
) -> dict[str, Any]:
    retrieved_psgs = []
    for rank, (passage_idx, score) in enumerate(ranked_candidates[:final_top_k], start=1):
        passage = passages[passage_idx]
        retrieved_psgs.append(
            {
                "rank": rank,
                "score": score,
                "title": passage.title,
                "url": passage.url,
                "date": passage.date,
                "contents": passage.contents,
                "psg_id": passage.psg_id,
                "retrieval_type": retrieval_type,
                "reranked": reranked,
            }
        )

    return {
        "task": sample["task"],
        "sample_index": sample["sample_index"],
        "question": sample["raw"].get("question", ""),
        "query": sample["query"],
        "answers": sample["raw"].get("answers", []),
        "positive_reference": sample["raw"].get("positive_reference", []),
        "retrieved_psgs": retrieved_psgs,
    }


def main() -> None:
    args = parse_args()

    domainrag_root = args.domainrag_root.resolve()
    corpus_json_root = resolve_corpus_root(domainrag_root, args.corpus_json_root).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_eval_samples(domainrag_root, args.tasks, args.max_samples)
    if not samples:
        raise RuntimeError("No evaluation samples loaded.")

    print(f"[info] requested max samples: {args.max_samples}")
    print(f"[info] actual loaded samples: {len(samples)}")
    print(f"[info] tasks: {', '.join(args.tasks)}")
    print(f"[info] corpus root: {corpus_json_root}")

    merged_dataset_path = output_dir / "domainrag_eval_samples_merged.jsonl"
    save_jsonl(
        merged_dataset_path,
        [
            {
                "task": sample["task"],
                "sample_index": sample["sample_index"],
                "question": sample["raw"].get("question", ""),
                "query": sample["query"],
                "raw": sample["raw"],
            }
            for sample in samples
        ],
    )

    passages = load_corpus_passages(corpus_json_root)
    print(f"[info] total passages indexed: {len(passages)}")

    bm25_retriever: BM25Retriever | None = None
    dense_retriever: DenseRetriever | None = None

    if args.retriever in {"bm25", "hybrid"}:
        bm25_retriever = BM25Retriever(passages)

    if args.retriever in {"dense", "hybrid"}:
        dense_retriever = DenseRetriever(
            passages=passages,
            model_path=args.dense_model_path,
            batch_size=args.embed_batch_size,
            device=args.device,
        )

    queries = [sample["query"] for sample in samples]
    candidate_lists: list[list[tuple[int, float]]] = []
    for query in tqdm(queries, desc="Retrieve queries", unit="query"):
        if args.retriever == "bm25":
            assert bm25_retriever is not None
            candidate_lists.append(bm25_retriever.search(query, args.top_k))
        elif args.retriever == "dense":
            assert dense_retriever is not None
            candidate_lists.append(dense_retriever.search(query, args.top_k))
        else:
            assert bm25_retriever is not None
            assert dense_retriever is not None
            dense_candidates = dense_retriever.search(query, args.dense_top_k)
            bm25_candidates = bm25_retriever.search(query, args.bm25_top_k)
            candidate_lists.append(
                fuse_hybrid_candidates(
                    dense_candidates=dense_candidates,
                    bm25_candidates=bm25_candidates,
                    top_k=args.top_k,
                    rrf_k=args.rrf_k,
                )
            )

    rerank_enabled = bool(args.reranker_path.strip())
    ranked_lists = candidate_lists
    if rerank_enabled:
        ranked_lists = rerank_candidates(
            queries=queries,
            candidate_lists=candidate_lists,
            passages=passages,
            reranker_path=args.reranker_path.strip(),
            batch_size=args.rerank_batch_size,
            device=args.device,
        )

    results = []
    for sample, ranked in tqdm(
        list(zip(samples, ranked_lists)),
        desc="Assemble results",
        unit="query",
    ):
        results.append(
            materialize_result(
                sample=sample,
                ranked_candidates=ranked,
                passages=passages,
                final_top_k=args.final_top_k,
                retrieval_type=args.retriever,
                reranked=rerank_enabled,
            )
        )

    output_name = f"domainrag_{args.retriever}_top{args.top_k}_samples{len(samples)}"
    if rerank_enabled:
        output_name += "_reranked"
    output_path = output_dir / f"{output_name}.jsonl"
    save_jsonl(output_path, results)

    summary = {
        "domainrag_root": str(domainrag_root),
        "corpus_json_root": str(corpus_json_root),
        "output_path": str(output_path),
        "tasks": args.tasks,
        "requested_max_samples": args.max_samples,
        "actual_sample_count": len(samples),
        "retriever": args.retriever,
        "reranker_path": args.reranker_path,
        "top_k": args.top_k,
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
        "final_top_k": args.final_top_k,
        "total_passages": len(passages),
    }
    summary_path = output_dir / f"{output_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] retrieval pipeline finished.")
    print(f"[done] merged samples: {merged_dataset_path}")
    print(f"[done] retrieved results: {output_path}")
    print(f"[done] run summary: {summary_path}")


if __name__ == "__main__":
    main()
