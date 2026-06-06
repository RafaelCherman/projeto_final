import os
import numpy as np
import faiss
from metrics import (ResourceMonitor, load_embeddings, load_qrels,
                     load_query_embeddings, run_queries, save_result)

# ── Configuração ───────────────────────────────────────────────────────────────
DATASETS       = ["msmarco", "fiqa", "scifact"]
MODEL          = "minilm"
DIMENSIONS     = [384, 128, 64]
TECHNIQUES     = ["flat", "ivf", "ivfpq"]
RESOURCE_LEVEL = os.environ.get("RESOURCE_LEVEL", "100pct")

N_CELLS = 100
N_PROBE = 10
M_PQ    = 8


def build_index(embeddings, dimension, technique):
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

    if technique == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    elif technique == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        index     = faiss.IndexIVFFlat(quantizer, dimension, N_CELLS,
                                        faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = N_PROBE

    elif technique == "ivfpq":
        m         = M_PQ if dimension % M_PQ == 0 else dimension // 8
        quantizer = faiss.IndexFlatIP(dimension)
        index     = faiss.IndexIVFPQ(quantizer, dimension, N_CELLS, m, 8)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = N_PROBE

    return index


def run():
    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print(f"{'='*50}")

        qrels = load_qrels(dataset)

        for dimension in DIMENSIONS:
            embeddings, doc_ids         = load_embeddings(dataset, MODEL, dimension)
            embeddings                  = embeddings.astype("float32")
            query_ids, query_embeddings = load_query_embeddings(dataset, MODEL, dimension)

            for technique in TECHNIQUES:
                print(f"\n  [MiniLM | dim={dimension} | {technique}]")

                monitor = ResourceMonitor()
                monitor.start()

                print(f"    Construindo índice FAISS ({technique})...")
                index = build_index(embeddings, dimension, technique)

                def search(qvec, k, idx=index, ids=doc_ids):
                    q      = np.array([qvec], dtype="float32")
                    _, I   = idx.search(q, k)
                    return [ids[i] for i in I[0] if i != -1]

                metrics   = run_queries(search, query_embeddings, query_ids, qrels)
                monitor.stop()
                resources = monitor.summary()

                result = {
                    "algorithm":      "faiss",
                    "dataset":        dataset,
                    "model":          MODEL,
                    "dimension":      dimension,
                    "technique":      technique,
                    "resource_level": RESOURCE_LEVEL,
                    "metrics":        {**metrics, **resources},
                }

                save_result(result, "faiss", RESOURCE_LEVEL,
                            dataset, MODEL, dimension, technique)


if __name__ == "__main__":
    run()