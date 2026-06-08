# Projeto Final de Curso - Documentação

## Estrutura do Projeto

### `experiments/`

- **`metrics.py`** - Módulo compartilhado entre os algoritmos. Contém as funções de carregamento de dados, monitoramento de CPU e memória, cálculo de Recall@k e MRR, medição de latência e salvamento dos resultados em JSON.

- **`annoy_experiment.py`** - Script de experimento do algoritmo Annoy. Constrói o índice com estrutura de árvores aleatórias e executa as buscas sobre os três datasets nas três dimensões de embeddings.

- **`faiss_experiment.py`** - Script de experimento do algoritmo FAISS. Avalia três técnicas de indexação: busca exata (Flat), clustering (IVF) e clustering com compressão (IVF+PQ), sobre os três datasets e dimensões.

- **`hnsw_experiment.py`** - Script de experimento do algoritmo HNSW. Constrói o índice com estrutura de grafo hierárquico e executa as buscas sobre os três datasets nas três dimensões de embeddings.

---

### `indexes/`

Índices ANN pré-construídos para o dataset SciFact, utilizados pelo protótipo. Contém os arquivos `.ann` (Annoy), `.index` (FAISS) e `.bin` (HNSW) nas dimensões 384, 128 e 64.

---

### `results/`

JSONs gerados pelos experimentos, organizados por algoritmo e nível de recurso (`100pct` e `50pct`). Cada arquivo contém as métricas de relevância (Recall@k, MRR), latência (P50, P95, P99) e consumo de recursos (CPU e memória) de uma combinação específica de experimento.

---

### `results_consolidated/`

Três arquivos `.txt` um por algoritmo, reunindo todos os resultados de cada algoritmo em um único documento de fácil leitura.

---

### `results_readable/`

Versão legível dos resultados em arquivos `.txt` individuais por experimento, com as métricas formatadas em linguagem natural.

---

### `app.py`

Protótipo de interface web desenvolvido em Streamlit com duas abas: busca semântica interativa sobre o dataset SciFact utilizando índices pré-construídos, e dashboard de visualização dos resultados dos experimentos com gráficos e tabela comparativa.

---

### `build_indexes.py`

Script de pré-construção dos índices ANN para o protótipo. Deve ser executado localmente antes do deploy, gerando os arquivos de índice persistidos na pasta `indexes/`.

---

### `consolidate_results.py`

Lê todos os JSONs de resultados e consolida em três arquivos `.txt` um por algoritmo, salvos em `results_consolidated/`.

---

### `dataset_script.py`

Realiza o download e padronização dos três datasets utilizados na pesquisa: MS MARCO (com amostragem de 100k passagens), FiQA e SciFact. Salva corpus, queries e qrels em formato `.jsonl` e `.tsv`.

---

### `Dockerfile`

Define a imagem Docker utilizada nos experimentos. Instala apenas as dependências necessárias para indexação e busca (numpy, faiss-cpu, annoy, hnswlib, psutil), sem modelos de linguagem.

---

### `embeddings_corpus.py`

Gera os embeddings das passagens do corpus usando o modelo MiniLM. Aplica PCA nas dimensões 128 e 64, salva os objetos PCA treinados em `.pkl` para uso posterior nas queries, e persiste os embeddings em arquivos `.npy`.

---

### `embeddings_queries.py`

Gera os embeddings das queries usando o modelo MiniLM. Carrega os objetos PCA salvos pelo `embeddings_corpus.py` e aplica a mesma transformação nas queries, garantindo que corpus e queries estejam no mesmo espaço vetorial.

---

### `read_results.py`

Lê os JSONs de resultados e gera arquivos `.txt` individuais por experimento em formato legível, organizados por algoritmo na pasta `results_readable/`.

---

### `requirements.txt`

Lista de dependências Python do protótipo Streamlit: streamlit, numpy, pandas, annoy, faiss-cpu, hnswlib e plotly.

---

### `run_experiments.sh`

Script shell que orquestra a execução completa dos experimentos. Constrói a imagem Docker uma única vez e executa sequencialmente os containers para cada algoritmo nos dois níveis de recurso (100% e 50%), aplicando os limites de CPU e memória via flags do Docker.
