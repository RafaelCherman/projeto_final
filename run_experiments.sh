#!/bin/bash
set -e

export MSYS_NO_PATHCONV=1   # ← adicione essa linha

# ── Configuração ───────────────────────────────────────────────────────────────
DATA_DIR="$(pwd)/data"
RESULTS_DIR="$(pwd)/results"
ALGORITHMS=("annoy" "faiss" "hnsw")

# Recursos: 100% = 4 cores / 8GB | 50% = 2 cores / 4GB
declare -A CPU_LIMIT=( ["100pct"]="4" ["50pct"]="2" )
declare -A MEM_LIMIT=( ["100pct"]="8g" ["50pct"]="4g" )

mkdir -p "$RESULTS_DIR"

# ── Build da imagem ────────────────────────────────────────────────────────────
echo "Building Docker image..."
docker build -t ann-experiment .

# ── Execução ───────────────────────────────────────────────────────────────────
for algo in "${ALGORITHMS[@]}"; do
    for level in "100pct" "50pct"; do
        echo ""
        echo "=================================================="
        echo "Algoritmo: $algo | Recursos: $level"
        echo "=================================================="

        docker run --rm \
            --cpus="${CPU_LIMIT[$level]}" \
            --memory="${MEM_LIMIT[$level]}" \
            --memory-swap="${MEM_LIMIT[$level]}" \
            -e RESOURCE_LEVEL="$level" \
            -e OMP_NUM_THREADS="${CPU_LIMIT[$level]}" \
            -v "$DATA_DIR":/data:ro \
            -v "$RESULTS_DIR":/results \
            ann-experiment \
            python experiments/${algo}_experiment.py

        echo "Concluído: $algo / $level"
    done
done

echo ""
echo "Todos os experimentos concluídos."
echo "Resultados em: $RESULTS_DIR"