from datasets import load_dataset
import os
import json
import csv
import random

os.makedirs("./data/msmarco", exist_ok=True)
os.makedirs("./data/fiqa",    exist_ok=True)
os.makedirs("./data/scifact", exist_ok=True)


def save_corpus(corpus, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, content in corpus.items():
            text = content["text"] if isinstance(content, dict) else content
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")


def save_queries(queries, path):
    with open(path, "w", encoding="utf-8") as f:
        for query_id, text in queries.items():
            f.write(json.dumps({"query_id": query_id, "text": text}) + "\n")


def save_qrels(qrels, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for query_id, doc_id, relevance in qrels:
            writer.writerow([query_id, doc_id, relevance])


# ── MS MARCO ──────────────────────────────────────────────────────────────────
def download_msmarco():
    print("Baixando MS MARCO...")

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # Extrai passagens individuais (cada exemplo tem uma lista de passages)
    docs    = {}
    queries = {}
    qrels   = []

    for example in dataset:
        query_id = str(example["query_id"])
        queries[query_id] = example["query"]

        for i, passage in enumerate(example["passages"]["passage_text"]):
            is_selected = example["passages"]["is_selected"][i]
            doc_id = f"{query_id}_{i}"
            docs[doc_id] = passage

            if is_selected == 1:
                qrels.append((query_id, doc_id, 1))

    print(f"  Total de passagens: {len(docs)}")
    print(f"  Total de queries:   {len(queries)}")
    print(f"  Total de qrels:     {len(qrels)}")

    # Sampling — garante que docs relevantes estejam incluídos
    required_ids  = set(qrel[1] for qrel in qrels)
    remaining_ids = list(set(docs.keys()) - required_ids)

    target = min(100_000, len(docs))
    fill_n = max(0, target - len(required_ids))
    random.seed(42)
    fill        = random.sample(remaining_ids, fill_n)
    sampled_ids = list(required_ids) + fill

    sampled_corpus  = {id: docs[id] for id in sampled_ids}
    relevant_qids   = set(q[0] for q in qrels)
    sampled_queries = {qid: text for qid, text in queries.items()
                       if qid in relevant_qids}

    save_corpus (sampled_corpus,  "./data/msmarco/corpus.jsonl")
    save_queries(sampled_queries, "./data/msmarco/queries.jsonl")
    save_qrels  (qrels,           "./data/msmarco/qrels.tsv")

    print(f"MS MARCO salvo — {len(sampled_corpus)} docs, "
          f"{len(sampled_queries)} queries, {len(qrels)} qrels\n")


def download_fiqa():
    print("Baixando FiQA...")

    corpus_ds  = load_dataset("beir/fiqa",  "corpus",  split="corpus")
    queries_ds = load_dataset("beir/fiqa",  "queries", split="queries")
    qrels_ds   = load_dataset("mteb/fiqa",             split="test")

    corpus  = {row["_id"]: row["text"]   for row in corpus_ds}
    queries = {row["_id"]: row["text"]   for row in queries_ds}
    qrels   = [(str(row["query-id"]), str(row["corpus-id"]), int(row["score"]))
                for row in qrels_ds]

    save_corpus (corpus,  "./data/fiqa/corpus.jsonl")
    save_queries(queries, "./data/fiqa/queries.jsonl")
    save_qrels  (qrels,   "./data/fiqa/qrels.tsv")

    print(f"FiQA salvo — {len(corpus)} docs, "
          f"{len(queries)} queries, {len(qrels)} qrels\n")


def download_scifact():
    print("Baixando SciFact...")

    corpus_ds  = load_dataset("beir/scifact", "corpus",  split="corpus")
    queries_ds = load_dataset("beir/scifact", "queries", split="queries")
    qrels_ds   = load_dataset("mteb/scifact",            split="test")

    corpus  = {row["_id"]: row["text"]   for row in corpus_ds}
    queries = {row["_id"]: row["text"]   for row in queries_ds}
    qrels   = [(str(row["query-id"]), str(row["corpus-id"]), int(row["score"]))
                for row in qrels_ds]

    save_corpus (corpus,  "./data/scifact/corpus.jsonl")
    save_queries(queries, "./data/scifact/queries.jsonl")
    save_qrels  (qrels,   "./data/scifact/qrels.tsv")

    print(f"SciFact salvo — {len(corpus)} docs, "
          f"{len(queries)} queries, {len(qrels)} qrels\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_msmarco()
    download_fiqa()
    download_scifact()
    print("Todos os datasets salvos em ./data/")