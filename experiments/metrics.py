# experiments/metrics.py
import time
import threading
import psutil
import os
import json
import numpy as np


# ── Coleta de CPU e memória em thread paralela ─────────────────────────────────
class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval    = interval
        self.cpu_samples = []
        self.mem_samples = []
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            self.cpu_samples.append(process.cpu_percent(interval=None))
            self.mem_samples.append(process.memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)

    def summary(self):
        return {
            "cpu_pct_mean":   round(float(np.mean(self.cpu_samples)),  2) if self.cpu_samples else 0,
            "cpu_pct_peak":   round(float(np.max(self.cpu_samples)),   2) if self.cpu_samples else 0,
            "memory_mb_mean": round(float(np.mean(self.mem_samples)),  2) if self.mem_samples else 0,
            "memory_mb_peak": round(float(np.max(self.mem_samples)),   2) if self.mem_samples else 0,
        }


# ── Carregamento de queries e qrels ───────────────────────────────────────────
def load_queries(dataset_name: str):
    path = f"/data/{dataset_name}/queries.jsonl"
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]
    return queries


def load_qrels(dataset_name: str):
    path = f"/data/{dataset_name}/qrels.tsv"
    qrels = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # pula header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, rel = parts[0], parts[1], int(parts[2])
            if rel > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


# ── Carregamento de embeddings do corpus ───────────────────────────────────────
def load_embeddings(dataset_name: str, model_name: str, dimension: int):
    if dimension in (128, 64):
        emb_file = f"embeddings_{model_name}_pca{dimension}.npy"
    else:
        emb_file = f"embeddings_{model_name}.npy"

    ids_file = f"embedding_ids_{model_name}.jsonl"

    embeddings = np.load(f"/data/{dataset_name}/{emb_file}")

    doc_ids = []
    with open(f"/data/{dataset_name}/{ids_file}", encoding="utf-8") as f:
        for line in f:
            doc_ids.append(json.loads(line)["doc_id"])

    return embeddings, doc_ids


# ── Carregamento de embeddings das queries ─────────────────────────────────────
def load_query_embeddings(dataset_name: str, model_name: str, dimension: int):
    emb_file = f"query_embeddings_{model_name}_{dimension}.npy"
    ids_file = f"query_ids_{model_name}.jsonl"

    embeddings = np.load(f"/data/{dataset_name}/{emb_file}")

    query_ids = []
    with open(f"/data/{dataset_name}/{ids_file}", encoding="utf-8") as f:
        for line in f:
            query_ids.append(json.loads(line)["query_id"])

    return query_ids, embeddings.astype("float32")


# ── Métricas de relevância ─────────────────────────────────────────────────────
def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int):
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: set):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ── Execução das queries e coleta de latência ──────────────────────────────────
def run_queries(search_fn, query_embeddings, query_ids, qrels, k=10):
    latencies  = []
    recalls_1  = []
    recalls_5  = []
    recalls_10 = []
    mrrs       = []

    for qid, qvec in zip(query_ids, query_embeddings):
        if qid not in qrels:
            continue

        relevant = qrels[qid]

        start     = time.perf_counter()
        retrieved = search_fn(qvec, k)
        elapsed   = (time.perf_counter() - start) * 1000

        latencies.append(elapsed)
        recalls_1.append(recall_at_k(retrieved, relevant, 1))
        recalls_5.append(recall_at_k(retrieved, relevant, 5))
        recalls_10.append(recall_at_k(retrieved, relevant, 10))
        mrrs.append(mean_reciprocal_rank(retrieved, relevant))

    latencies = np.array(latencies)

    return {
        "recall_at_1":    round(float(np.mean(recalls_1)),            4),
        "recall_at_5":    round(float(np.mean(recalls_5)),            4),
        "recall_at_10":   round(float(np.mean(recalls_10)),           4),
        "mrr":            round(float(np.mean(mrrs)),                  4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)),  2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)),  2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)),  2),
        "num_queries":    len(latencies),
    }


# ── Salvamento do resultado ────────────────────────────────────────────────────
def save_result(result: dict, algorithm: str, resource_level: str,
                dataset: str, model: str, dimension: int, technique: str):

    out_dir  = f"/results/{algorithm}/{resource_level}"
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{dataset}_{model}_{dimension}_{technique}.json"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  Resultado salvo em {filepath}")