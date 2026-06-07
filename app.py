import streamlit as st
import numpy as np
import json
import os
import random
import time
import pandas as pd
import plotly.express as px

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANN Semantic Search",
    layout="wide"
)

# ── Constantes ─────────────────────────────────────────────────────────────────
DATA_DIR    = "./data/scifact"
INDEXES_DIR = "./indexes/scifact"
RESULTS_DIR = "./results"
ALGORITHMS  = ["annoy", "faiss", "hnsw"]
DIMENSIONS  = [384, 128, 64]
TECHNIQUES  = {
    "annoy": ["flat"],
    "faiss": ["flat", "ivf", "ivfpq"],
    "hnsw":  ["flat"],
}

# ── Carregamento de dados (cache) ──────────────────────────────────────────────
@st.cache_resource
def load_corpus():
    corpus = {}
    with open(f"{DATA_DIR}/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["doc_id"]] = row["text"]
    return corpus


@st.cache_resource
def load_queries():
    queries = {}
    with open(f"{DATA_DIR}/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["text"]
    return queries


@st.cache_resource
def load_qrels():
    qrels = {}
    with open(f"{DATA_DIR}/qrels.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, rel = parts[0], parts[1], int(parts[2])
            if rel > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


@st.cache_resource
def load_embeddings(dimension: int):
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


@st.cache_resource
def load_query_embeddings(dimension: int):
    emb_file   = f"query_embeddings_minilm_{dimension}.npy"
    embeddings = np.load(f"{DATA_DIR}/{emb_file}")

    query_ids = []
    with open(f"{DATA_DIR}/query_ids_minilm.jsonl", encoding="utf-8") as f:
        for line in f:
            query_ids.append(json.loads(line)["query_id"])

    return {qid: emb for qid, emb in zip(query_ids, embeddings)}


# ── Carregamento de índices pré-construídos (cache) ────────────────────────────
@st.cache_resource
def load_annoy_index(dimension: int):
    from annoy import AnnoyIndex
    index = AnnoyIndex(dimension, "angular")
    index.load(f"{INDEXES_DIR}/annoy_{dimension}.ann")
    return index


@st.cache_resource
def load_faiss_index(dimension: int, technique: str):
    import faiss
    index = faiss.read_index(f"{INDEXES_DIR}/faiss_{technique}_{dimension}.index")
    if technique in ("ivf", "ivfpq"):
        index.nprobe = 10
    return index


@st.cache_resource
def load_hnsw_index(dimension: int):
    import hnswlib
    _, doc_ids = load_embeddings(dimension)
    index      = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(
        f"{INDEXES_DIR}/hnsw_{dimension}.bin",
        max_elements=len(doc_ids)
    )
    index.set_ef(50)
    return index


# ── Carregamento de resultados dos experimentos (cache) ────────────────────────
@st.cache_resource
def load_all_results():
    results = []
    for algo in ALGORITHMS:
        for level in ["100pct", "50pct"]:
            level_dir = os.path.join(RESULTS_DIR, algo, level)
            if not os.path.exists(level_dir):
                continue
            for filename in os.listdir(level_dir):
                if not filename.endswith(".json"):
                    continue
                with open(os.path.join(level_dir, filename), encoding="utf-8") as f:
                    data = json.load(f)
                    results.append({
                        "Algoritmo": data["algorithm"].upper(),
                        "Dataset":   data["dataset"].upper(),
                        "Dimensão":  data["dimension"],
                        "Técnica":   data["technique"].upper(),
                        "Recursos":  data["resource_level"],
                        "Recall@1":  round(data["metrics"]["recall_at_1"]  * 100, 2),
                        "Recall@5":  round(data["metrics"]["recall_at_5"]  * 100, 2),
                        "Recall@10": round(data["metrics"]["recall_at_10"] * 100, 2),
                        "MRR":       round(data["metrics"]["mrr"]          * 100, 2),
                        "P50 (ms)":  data["metrics"]["latency_p50_ms"],
                        "P95 (ms)":  data["metrics"]["latency_p95_ms"],
                        "P99 (ms)":  data["metrics"]["latency_p99_ms"],
                        "CPU média": data["metrics"]["cpu_pct_mean"],
                        "CPU pico":  data["metrics"]["cpu_pct_peak"],
                        "Mem média": data["metrics"]["memory_mb_mean"],
                        "Mem pico":  data["metrics"]["memory_mb_peak"],
                    })
    return pd.DataFrame(results)


# ── Função de busca ────────────────────────────────────────────────────────────
def run_search(algorithm, technique, dimension, query_id,
               query_embeddings_map, doc_ids):
    query_vec = query_embeddings_map[query_id]

    start = time.perf_counter()

    if algorithm == "annoy":
        index     = load_annoy_index(dimension)
        indices   = index.get_nns_by_vector(query_vec.tolist(), 10)
        retrieved = [doc_ids[i] for i in indices]

    elif algorithm == "faiss":
        index     = load_faiss_index(dimension, technique)
        q         = np.array([query_vec], dtype="float32")
        _, I      = index.search(q, 10)
        retrieved = [doc_ids[i] for i in I[0] if i != -1]

    elif algorithm == "hnsw":
        index     = load_hnsw_index(dimension)
        labels, _ = index.knn_query(np.array([query_vec]), k=10)
        retrieved = [doc_ids[i] for i in labels[0]]

    latency = (time.perf_counter() - start) * 1000
    return retrieved, latency


# ── Abas ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Busca Interativa", "Dashboard de Resultados"])


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — BUSCA INTERATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Busca Semântica Interativa")
    st.markdown("Dataset: **SciFact** — artigos científicos com afirmações e evidências.")

    corpus  = load_corpus()
    queries = load_queries()
    qrels   = load_qrels()

    # Query aleatória
    if "current_query_id" not in st.session_state:
        st.session_state.current_query_id = random.choice(list(queries.keys()))

    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("&nbsp;")
        if st.button("Nova query", use_container_width=True):
            st.session_state.current_query_id = random.choice(list(queries.keys()))
            st.rerun()
    with col2:
        st.markdown("**Query selecionada:**")
        st.info(queries[st.session_state.current_query_id])
        

    st.divider()

    # Configuração
    st.subheader("Configuração")
    col1, col2, col3 = st.columns(3)

    with col1:
        algorithm = st.selectbox("Algoritmo", ALGORITHMS, format_func=str.upper)
    with col2:
        dimension = st.selectbox("Dimensão",  DIMENSIONS)
    with col3:
        technique = st.selectbox("Técnica", TECHNIQUES[algorithm], format_func=str.upper)

    st.divider()

    # Busca
    if st.button("Realizar Busca", type="primary", use_container_width=True):
        current_qid = st.session_state.current_query_id

        _, doc_ids           = load_embeddings(dimension)
        query_embeddings_map = load_query_embeddings(dimension)

        if current_qid not in query_embeddings_map:
            st.error("Query não encontrada nos embeddings gerados.")
        else:
            with st.spinner("Realizando busca..."):
                retrieved, latency = run_search(
                    algorithm, technique, dimension,
                    current_qid, query_embeddings_map, doc_ids
                )

            relevant = qrels.get(current_qid, set())
            if not relevant:
                st.warning(
                    "Esta query não possui documentos relevantes anotados no gabarito. "
                    "Os resultados abaixo são os documentos mais próximos semanticamente, "
                    "mas não é possível avaliar se são corretos."
                )
            hits     = sum(1 for d in retrieved if d in relevant)

            # Posição do primeiro documento relevante
            first_hit_position = None
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    first_hit_position = rank
                    break

            # Mensagem de resultado em linguagem natural
            st.subheader("Resultado da Busca")

            if first_hit_position is not None:
                st.success(
                    f"✅ O documento esperado foi encontrado na posição "
                    f"**#{first_hit_position}** de 10 resultados — "
                    f"Busca concluída em **{latency:.2f}ms**"
                )
            else:
                st.error(
                    f"❌ O documento esperado não apareceu nos 10 resultados — "
                    f"Busca concluída em **{latency:.2f}ms**"
                )

            st.divider()

            # Resultados
            st.subheader("Documentos Retornados")
            for rank, doc_id in enumerate(retrieved, start=1):
                is_relevant = doc_id in relevant
                text        = corpus.get(doc_id, "Documento não encontrado.")
                preview     = text[:300] + "..." if len(text) > 300 else text

                if is_relevant:
                    st.success(f"**#{rank} ✅ Relevante** — `{doc_id}`\n\n{preview}")
                else:
                    st.error(f"**#{rank} ❌ Não relevante** — `{doc_id}`\n\n{preview}")
            
          


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Dashboard de Resultados")

    df = load_all_results()

    if df.empty:
        st.warning("Nenhum resultado encontrado em ./results/")
    else:
        # Filtros
        st.subheader("Filtros")
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_dataset = st.multiselect(
                "Dataset",
                options=df["Dataset"].unique().tolist(),
                default=df["Dataset"].unique().tolist()
            )
        with col2:
            filter_recursos = st.multiselect(
                "Recursos",
                options=df["Recursos"].unique().tolist(),
                default=df["Recursos"].unique().tolist()
            )
        with col3:
            filter_algo = st.multiselect(
                "Algoritmo",
                options=df["Algoritmo"].unique().tolist(),
                default=df["Algoritmo"].unique().tolist()
            )

        df_filtered = df[
            df["Dataset"].isin(filter_dataset) &
            df["Recursos"].isin(filter_recursos) &
            df["Algoritmo"].isin(filter_algo)
        ]

        st.divider()

        # Gráfico Recall@10
        st.subheader("Recall@10 por Algoritmo e Dimensão")

        recall_df = df_filtered.groupby(
            ["Algoritmo", "Dimensão"], as_index=False
        )["Recall@10"].mean()

        fig_recall = px.bar(
            recall_df,
            x="Dimensão",
            y="Recall@10",
            color="Algoritmo",
            barmode="group",
            text_auto=".1f",
            labels={"Recall@10": "Recall@10 (%)", "Dimensão": "Dimensão"},
            category_orders={"Dimensão": [64, 128, 384]},
            height=450,
        )
        
        fig_recall.update_layout(
            bargap=0.4,      # Diminui o espaçamento geral entre os grupos (64, 128, 384)
            bargroupgap=0.1, # Diminui o espaçamento entre as colunas do mesmo grupo
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        
        # FORÇA o eixo X a ser tratado como categoria, removendo 192, 256, 320
        fig_recall.update_xaxes(type='category')
        
        # REMOVA A LINHA: fig_recall.update_traces(width=0.2)
        # Ao remover a limitação de width, o Plotly usará a grossura ideal baseada no bargap

        st.plotly_chart(fig_recall, use_container_width=True)

        st.divider()

        # Gráfico Latência
        st.subheader("Latência P50 por Algoritmo e Técnica")

        latency_df = df_filtered.groupby(
            ["Algoritmo", "Técnica"], as_index=False
        )["P50 (ms)"].mean()

        fig_latency = px.bar(
            latency_df,
            x="Técnica",
            y="P50 (ms)",
            color="Algoritmo",
            barmode="group",
            text_auto=".2f",
            labels={"P50 (ms)": "Latência P50 (ms)", "Técnica": "Técnica"},
            category_orders={"Técnica": ["FLAT", "IVF", "IVFPQ"]},
            height=450,
        )
        fig_latency.update_layout(
            bargap=0.3,
            bargroupgap=0.1,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        fig_latency.update_traces(width=0.2)
        st.plotly_chart(fig_latency, use_container_width=True)

        st.divider()

        # Tabela completa
        st.subheader("Tabela Completa de Resultados")
        st.dataframe(
            df_filtered.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Baixar CSV",
            data=csv,
            file_name="resultados_ann.csv",
            mime="text/csv",
            use_container_width=True
        )