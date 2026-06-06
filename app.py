import streamlit as st
import subprocess
import json
import os

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANN Experiments",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 ANN Experiment Runner")
st.markdown("Configure e execute experimentos de busca semântica com algoritmos ANN.")

# ── Opções disponíveis ─────────────────────────────────────────────────────────
ALGORITHMS = ["annoy", "faiss", "hnsw"]
DATASETS   = ["msmarco", "fiqa", "scifact"]
DIMENSIONS = [384, 128, 64]
TECHNIQUES = {
    "annoy": ["flat"],
    "faiss": ["flat", "ivf", "ivfpq"],
    "hnsw":  ["flat"],
}
RESOURCE_LEVELS = {
    "100% (4 cores / 8GB)": {"level": "100pct", "cpus": "4", "memory": "8g"},
    "50% (2 cores / 4GB)":  {"level": "50pct",  "cpus": "2", "memory": "4g"},
}

# ── Formulário de seleção ──────────────────────────────────────────────────────
st.subheader("Configuração do Experimento")

col1, col2 = st.columns(2)

with col1:
    algorithm = st.selectbox("Algoritmo",  ALGORITHMS)
    dataset   = st.selectbox("Dataset",    DATASETS)
    dimension = st.selectbox("Dimensão",   DIMENSIONS)

with col2:
    technique      = st.selectbox("Técnica", TECHNIQUES[algorithm])
    resource_label = st.selectbox("Recursos", list(RESOURCE_LEVELS.keys()))

resource = RESOURCE_LEVELS[resource_label]

# Nome do arquivo de resultado esperado
result_filename = f"{dataset}_minilm_{dimension}_{technique}.json"
result_path     = os.path.join(
    "./results", algorithm, resource["level"], result_filename
)

st.divider()

# ── Resumo da configuração ─────────────────────────────────────────────────────
st.subheader("Resumo")
st.markdown(f"""
- **Algoritmo:** `{algorithm.upper()}`
- **Dataset:** `{dataset.upper()}`
- **Dimensão:** `{dimension}`
- **Técnica:** `{technique.upper()}`
- **Recursos:** `{resource_label}`
- **Arquivo de resultado:** `{result_path}`
""")

st.divider()

# ── Execução ───────────────────────────────────────────────────────────────────
st.subheader("Execução")

if st.button("▶ Rodar Experimento", type="primary", use_container_width=True):

    # Verifica se o resultado já existe
    if os.path.exists(result_path):
        st.warning(f"Resultado já existe em `{result_path}`. Sobrescrevendo...")

    # Monta o comando docker run
    data_dir    = os.path.abspath("./data")
    results_dir = os.path.abspath("./results")

    os.makedirs(os.path.join(results_dir, algorithm, resource["level"]), exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        f"--cpus={resource['cpus']}",
        f"--memory={resource['memory']}",
        f"--memory-swap={resource['memory']}",
        "-e", f"RESOURCE_LEVEL={resource['level']}",
        "-e", f"OMP_NUM_THREADS={resource['cpus']}",
        # Filtra apenas o dataset e configuração selecionados
        "-e", f"SINGLE_DATASET={dataset}",
        "-e", f"SINGLE_DIMENSION={dimension}",
        "-e", f"SINGLE_TECHNIQUE={technique}",
        "-v", f"{data_dir}:/data:ro",
        "-v", f"{results_dir}:/results",
        "ann-experiment",
        "python", f"experiments/{algorithm}_experiment.py"
    ]

    with st.spinner(f"Executando experimento — {algorithm.upper()} | {dataset.upper()} | dim={dimension} | {technique.upper()}..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hora de timeout
            )

            # Output do container
            if result.stdout:
                with st.expander("📋 Log do container", expanded=False):
                    st.code(result.stdout, language="bash")

            if result.returncode != 0:
                st.error("❌ Erro na execução do experimento.")
                if result.stderr:
                    st.code(result.stderr, language="bash")

            else:
                st.success("✅ Experimento concluído com sucesso!")

        except subprocess.TimeoutExpired:
            st.error("❌ Timeout — o experimento demorou mais de 1 hora.")
        except FileNotFoundError:
            st.error("❌ Docker não encontrado. Verifique se o Docker Desktop está rodando.")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")

st.divider()

# ── Exibição do resultado ──────────────────────────────────────────────────────
st.subheader("Resultado")

if os.path.exists(result_path):
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    m = data["metrics"]

    # Métricas de relevância
    st.markdown("**Relevância**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall@1",  f"{m['recall_at_1']  * 100:.2f}%")
    col2.metric("Recall@5",  f"{m['recall_at_5']  * 100:.2f}%")
    col3.metric("Recall@10", f"{m['recall_at_10'] * 100:.2f}%")
    col4.metric("MRR",       f"{m['mrr']          * 100:.2f}%")

    # Métricas de latência
    st.markdown("**Latência**")
    col1, col2, col3 = st.columns(3)
    col1.metric("P50", f"{m['latency_p50_ms']} ms")
    col2.metric("P95", f"{m['latency_p95_ms']} ms")
    col3.metric("P99", f"{m['latency_p99_ms']} ms")

    # Métricas de recursos
    st.markdown("**Recursos**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU média",     f"{m['cpu_pct_mean']}%")
    col2.metric("CPU pico",      f"{m['cpu_pct_peak']}%")
    col3.metric("Memória média", f"{m['memory_mb_mean']} MB")
    col4.metric("Memória pico",  f"{m['memory_mb_peak']} MB")

    # JSON completo
    st.markdown("**JSON completo**")
    st.json(data)

    # Download do JSON
    st.download_button(
        label="⬇ Baixar JSON",
        data=json.dumps(data, indent=2),
        file_name=result_filename,
        mime="application/json",
        use_container_width=True
    )

else:
    st.info("Nenhum resultado encontrado para essa configuração. Execute o experimento acima.")