# build_indexes.py
import numpy as np
import json
import os
from annoy import AnnoyIndex
import faiss
import hnswlib

DATA_DIR    = "./data/scifact"
INDEXES_DIR = "./indexes/scifact"
DIMENSIONS  = [384, 128, 64]

os.makedirs(INDEXES_DIR, exist_ok=True)


def load_embeddings(dimension):
    if dimension in (128, 64):
        emb_file = f"embeddings_minilm_pca{dimension}.npy"
    else:
        emb_file = "embeddings_minilm.npy"

    embeddings = np.load(f"{DATA_DIR}/{emb_file}")

    doc_ids = []
    with open(f"{DATA_DIR}/embedding_ids_minilm.jsonl", encoding="utf-8") as f:
        for line in f:
            doc_ids.append(json.loads(line)["doc_id"])

    return embeddings, doc_ids


for dim in DIMENSIONS:
    print(f"\nConstruindo índices para dimensão {dim}...")
    embeddings, doc_ids = load_embeddings(dim)
    n = len(embeddings)

    # ── Annoy ──────────────────────────────────────────────────────────────────
    print(f"  Annoy (n_trees=100)...")
    annoy_index = AnnoyIndex(dim, "angular")
    for i, vec in enumerate(embeddings):
        annoy_index.add_item(i, vec.tolist())
    annoy_index.build(100)
    annoy_index.save(f"{INDEXES_DIR}/annoy_{dim}.ann")
    print(f"  Salvo: annoy_{dim}.ann")

    # ── FAISS Flat ─────────────────────────────────────────────────────────────
    print(f"  FAISS Flat...")
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embeddings.astype("float32"))
    faiss.write_index(flat_index, f"{INDEXES_DIR}/faiss_flat_{dim}.index")
    print(f"  Salvo: faiss_flat_{dim}.index")

    # ── FAISS IVF ──────────────────────────────────────────────────────────────
    print(f"  FAISS IVF...")
    quantizer = faiss.IndexFlatIP(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
    ivf_index.train(embeddings.astype("float32"))
    ivf_index.add(embeddings.astype("float32"))
    ivf_index.nprobe = 10
    faiss.write_index(ivf_index, f"{INDEXES_DIR}/faiss_ivf_{dim}.index")
    print(f"  Salvo: faiss_ivf_{dim}.index")

    # ── FAISS IVFPQ ────────────────────────────────────────────────────────────
    print(f"  FAISS IVFPQ...")
    m = 8 if dim % 8 == 0 else dim // 8
    quantizer = faiss.IndexFlatIP(dim)
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dim, 100, m, 8)
    ivfpq_index.train(embeddings.astype("float32"))
    ivfpq_index.add(embeddings.astype("float32"))
    ivfpq_index.nprobe = 10
    faiss.write_index(ivfpq_index, f"{INDEXES_DIR}/faiss_ivfpq_{dim}.index")
    print(f"  Salvo: faiss_ivfpq_{dim}.index")

    # ── HNSW ───────────────────────────────────────────────────────────────────
    print(f"  HNSW (M=16, ef_construction=200)...")
    hnsw_index = hnswlib.Index(space="cosine", dim=dim)
    hnsw_index.init_index(max_elements=n, ef_construction=200, M=16)
    hnsw_index.add_items(embeddings, list(range(n)))
    hnsw_index.set_ef(50)
    hnsw_index.save_index(f"{INDEXES_DIR}/hnsw_{dim}.bin")
    print(f"  Salvo: hnsw_{dim}.bin")

print("\nTodos os índices construídos.")