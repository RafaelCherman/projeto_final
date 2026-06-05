from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
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
    return len(text.strip().split()) >= 10


# ── Leitura do corpus ──────────────────────────────────────────────────────────
def load_corpus(dataset_name: str):
    path = f"./data/{dataset_name}/corpus.jsonl"
    doc_ids, texts = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row  = json.loads(line)
            text = row["text"].strip()

            if not is_valid(text):
                continue

            if dataset_name == "fiqa":
                text = re.sub(r"http\S+", "", text).strip()
                if not is_valid(text):
                    continue

            doc_ids.append(row["doc_id"])
            texts.append(text)

    return doc_ids, texts


# ── Geração de embeddings ──────────────────────────────────────────────────────
def generate_embeddings(dataset_name: str):
    print(f"Processando {dataset_name}...")

    doc_ids, texts = load_corpus(dataset_name)
    print(f"  Passagens válidas: {len(texts)}")

    out_dir = f"./data/{dataset_name}"

    # Gera embeddings originais (384)
    embeddings_384 = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Salva embeddings originais
    np.save(f"{out_dir}/embeddings_{MODEL_KEY}.npy", embeddings_384)
    print(f"  Salvo: embeddings_{MODEL_KEY}.npy — shape {embeddings_384.shape}")

    # Salva IDs
    with open(f"{out_dir}/embedding_ids_{MODEL_KEY}.jsonl", "w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(json.dumps({"doc_id": doc_id}) + "\n")
    print(f"  Salvo: embedding_ids_{MODEL_KEY}.jsonl")

    # Gera, salva embeddings PCA e salva o objeto PCA em disco
    for dim in DIMENSIONS:
        print(f"\n  Aplicando PCA {dim}...")
        pca     = PCA(n_components=dim, random_state=42)
        reduced = pca.fit_transform(embeddings_384)

        variance = pca.explained_variance_ratio_.sum() * 100
        print(f"  Variância preservada: {variance:.1f}%")

        # Normaliza após PCA
        norms           = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reduced         = reduced / norms

        # Salva embeddings reduzidos
        np.save(f"{out_dir}/embeddings_{MODEL_KEY}_pca{dim}.npy", reduced)
        print(f"  Salvo: embeddings_{MODEL_KEY}_pca{dim}.npy — shape {reduced.shape}")

        # Salva objeto PCA para uso nas queries
        joblib.dump(pca, f"{out_dir}/pca_{MODEL_KEY}_{dim}.pkl")
        print(f"  Salvo: pca_{MODEL_KEY}_{dim}.pkl")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for dataset in DATASETS:
        generate_embeddings(dataset)

    print("Embeddings do corpus gerados para todos os datasets.")