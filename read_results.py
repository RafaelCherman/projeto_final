# read_results.py
import json
import os

# ── Configuração ───────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
OUTPUT_DIR  = "./results_readable"
ALGORITHMS  = ["annoy", "faiss", "hnsw"]


# ── Formatação de um experimento ───────────────────────────────────────────────
def format_result(data: dict) -> str:
    m = data["metrics"]

    max_cpu = 200 if data["resource_level"] == "50pct" else 400

    cpu_uso_medio = (m['cpu_pct_mean'] / max_cpu) * 100
    cpu_uso_pico  = (m['cpu_pct_peak'] / max_cpu) * 100

    lines = [
        f"Dataset:        {data['dataset'].upper()}",
        f"Modelo:         {data['model'].upper()}",
        f"Dimensão:       {data['dimension']}",
        f"Técnica:        {data['technique'].upper()}",
        f"Recursos:       {data['resource_level']}",
        f"",
        f"── Relevância ──────────────────────────────",
        f"Recall@1:       {m['recall_at_1'] * 100:.2f}%",
        f"Recall@5:       {m['recall_at_5'] * 100:.2f}%",
        f"Recall@10:      {m['recall_at_10'] * 100:.2f}%",
        f"MRR:            {m['mrr'] * 100:.2f}%",
        f"Queries:        {m['num_queries']}",
        f"",
        f"── Latência ────────────────────────────────",
        f"P50:            {m['latency_p50_ms']} ms",
        f"P95:            {m['latency_p95_ms']} ms",
        f"P99:            {m['latency_p99_ms']} ms",
        f"",
        f"── Recursos ────────────────────────────────",
        f"CPU média:      {m['cpu_pct_mean']}% ({cpu_uso_medio:.1f}% do limite)",
        f"CPU pico:       {m['cpu_pct_peak']}% ({cpu_uso_pico:.1f}% do limite)",
        f"Memória média:  {m['memory_mb_mean']} MB",
        f"Memória pico:   {m['memory_mb_peak']} MB",
    ]

    return "\n".join(lines)


# ── Leitura e escrita ──────────────────────────────────────────────────────────
def process_algorithm(algorithm: str):
    algo_input_dir  = os.path.join(RESULTS_DIR, algorithm)
    algo_output_dir = os.path.join(OUTPUT_DIR,  algorithm)

    if not os.path.exists(algo_input_dir):
        print(f"  Pasta não encontrada, pulando: {algo_input_dir}")
        return

    os.makedirs(algo_output_dir, exist_ok=True)

    json_files = []
    for level in ["100pct", "50pct"]:
        level_dir = os.path.join(algo_input_dir, level)
        if not os.path.exists(level_dir):
            continue
        for filename in sorted(os.listdir(level_dir)):
            if filename.endswith(".json"):
                json_files.append(os.path.join(level_dir, filename))

    if not json_files:
        print(f"  Nenhum JSON encontrado para {algorithm}")
        return

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Nome do arquivo de saída espelha o JSON original
        relative = os.path.relpath(json_path, os.path.join(RESULTS_DIR, algorithm))
        txt_name = relative.replace(os.sep, "_").replace(".json", ".txt")
        txt_path = os.path.join(algo_output_dir, txt_name)

        content = format_result(data)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  Salvo: {txt_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for algo in ALGORITHMS:
        print(f"\n[{algo.upper()}]")
        process_algorithm(algo)

    print("\nConcluído. Resultados legíveis em:", OUTPUT_DIR)