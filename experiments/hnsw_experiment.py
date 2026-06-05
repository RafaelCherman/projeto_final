import os
import numpy as np
import hnswlib
from metrics import (ResourceMonitor, load_embeddings, load_qrels,
                     load_query_embeddings, run_queries, save_result)

# ── Configuração ───────────────────────────────────────────────────────────────
DATASETS       = ["msmarco", "fiqa", "scifact"]
MODEL          = "minilm"
DIMENSIONS     = [384, 128, 64]
RESOURCE_LEVEL = os.environ.get("RESOURCE_LEVEL", "100pct")

M               = 16
EF_CONSTRUCTION = 200
EF_SEARCH       = 50


def build_and_search(embeddings, doc_ids, dimension, query_embeddings, query_ids, qrels):
    n = len(embeddings)

    print(f"    Construindo índice HNSW (M={M}, ef_construction={EF_CONSTRUCTION})...")
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)
    index.add_items(embeddings, list(range(n)))
    index.set_ef(EF_SEARCH)

    def search(qvec, k):
        labels, _ = index.knn_query(np.array([qvec]), k=k)
        return [doc_ids[i] for i in labels[0]]

    return run_queries(search, query_embeddings, query_ids, qrels)


def run():
    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print(f"{'='*50}")

        qrels = load_qrels(dataset)

        for dimension in DIMENSIONS:
            print(f"\n  [MiniLM | dim={dimension} | flat]")

            embeddings, doc_ids         = load_embeddings(dataset, MODEL, dimension)
            query_ids, query_embeddings = load_query_embeddings(dataset, MODEL, dimension)

            monitor = ResourceMonitor()
            monitor.start()

            metrics = build_and_search(
                embeddings, doc_ids, dimension,
                query_embeddings, query_ids, qrels
            )

            monitor.stop()
            resources = monitor.summary()

            result = {
                "algorithm":      "hnsw",
                "dataset":        dataset,
                "model":          MODEL,
                "dimension":      dimension,
                "technique":      "flat",
                "resource_level": RESOURCE_LEVEL,
                "metrics":        {**metrics, **resources},
            }

            save_result(result, "hnsw", RESOURCE_LEVEL,
                        dataset, MODEL, dimension, "flat")


if __name__ == "__main__":
    run()