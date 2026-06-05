from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
import os
import joblib

# ── Configuração ───────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_KEY  = "minilm"
BATCH_SIZE = 64
DATASETS   = ["msmarco", "fiqa", "scifact"]
DIMENSIONS = [128, 64]

model = SentenceTransformer(MODEL_NAME)
print(f"Modelo carregado: {MODEL_NAME}")
print(f"Dimensão original: {model.get_sentence_embedding_dimension()}\n")


# ── Limpeza mínima ─────────────────────────────────────────────────────────────
def is_valid(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return len(text.strip().split()) >= 3


# ── Leitura das queries ────────────────────────────────────────────────────────
def load_queries(dataset_name: str):
    path = f"./data/{dataset_name}/queries.jsonl"
    query_ids, texts = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row  = json.loads(line)
            text = row["text"].strip()

            if dataset_name == "fiqa":
                text = re.sub(r"http\S+", "", text).strip()

            if not is_valid(text):
                continue

            query_ids.append(row["query_id"])
            texts.append(text)

    return query_ids, texts


# ── Geração de embeddings ──────────────────────────────────────────────────────
def generate_query_embeddings(dataset_name: str):
    print(f"Processando {dataset_name}...")

    query_ids, texts = load_queries(dataset_name)
    print(f"  Queries válidas: {len(texts)}")

    out_dir = f"./data/{dataset_name}"

    # Gera embeddings originais (384)
    embeddings_384 = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Salva IDs — único arquivo para todas as dimensões
    with open(f"{out_dir}/query_ids_{MODEL_KEY}.jsonl", "w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(json.dumps({"query_id": qid}) + "\n")
    print(f"  Salvo: query_ids_{MODEL_KEY}.jsonl")

    # Salva embeddings originais (384)
    np.save(f"{out_dir}/query_embeddings_{MODEL_KEY}_384.npy", embeddings_384)
    print(f"  Salvo: query_embeddings_{MODEL_KEY}_384.npy — shape {embeddings_384.shape}")

    # Aplica o PCA do corpus nas queries
    for dim in DIMENSIONS:
        pkl_path = f"{out_dir}/pca_{MODEL_KEY}_{dim}.pkl"

        if not os.path.exists(pkl_path):
            print(f"  PCA não encontrado, pulando dim {dim}: {pkl_path}")
            continue

        print(f"\n  Aplicando PCA {dim} do corpus...")
        pca     = joblib.load(pkl_path)
        reduced = pca.transform(embeddings_384)  # transform, não fit_transform

        # Normaliza após PCA
        norms           = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reduced         = reduced / norms

        np.save(f"{out_dir}/query_embeddings_{MODEL_KEY}_{dim}.npy", reduced)
        print(f"  Salvo: query_embeddings_{MODEL_KEY}_{dim}.npy — shape {reduced.shape}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for dataset in DATASETS:
        generate_query_embeddings(dataset)

    print("Query embeddings gerados para todos os datasets.")