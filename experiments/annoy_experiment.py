import os
import numpy as np
from annoy import AnnoyIndex
from metrics import (ResourceMonitor, load_embeddings, load_qrels,
                     load_query_embeddings, run_queries, save_result)

# ── Configuração ───────────────────────────────────────────────────────────────
DATASETS       = ["msmarco", "fiqa", "scifact"]
MODEL          = "minilm"
DIMENSIONS     = [384, 128, 64]
N_TREES        = 100
RESOURCE_LEVEL = os.environ.get("RESOURCE_LEVEL", "100pct")


def build_and_search(embeddings, doc_ids, dimension, query_embeddings, query_ids, qrels):
    print(f"    Construindo índice Annoy (n_trees={N_TREES})...")
    index = AnnoyIndex(dimension, "angular")

    for i, vec in enumerate(embeddings):
        index.add_item(i, vec.tolist())

    index.build(N_TREES)

    def search(qvec, k):
        indices = index.get_nns_by_vector(qvec.tolist(), k)
        return [doc_ids[i] for i in indices]

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
                "algorithm":      "annoy",
                "dataset":        dataset,
                "model":          MODEL,
                "dimension":      dimension,
                "technique":      "flat",
                "resource_level": RESOURCE_LEVEL,
                "metrics":        {**metrics, **resources},
            }

            save_result(result, "annoy", RESOURCE_LEVEL,
                        dataset, MODEL, dimension, "flat")


if __name__ == "__main__":
    run()