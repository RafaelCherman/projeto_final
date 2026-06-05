from datasets import load_dataset, get_dataset_config_names

print("=== FIQA ===")
print("beir/fiqa configs:", get_dataset_config_names("beir/fiqa"))

try:
    ds = load_dataset("mteb/fiqa", split="test")
    print("mteb/fiqa test colunas:", ds.column_names)
    print("mteb/fiqa test exemplo:", ds[0])
except Exception as e:
    print("mteb/fiqa erro:", e)

print("\n=== SCIFACT ===")
print("beir/scifact configs:", get_dataset_config_names("beir/scifact"))

try:
    ds = load_dataset("mteb/scifact", split="test")
    print("mteb/scifact test colunas:", ds.column_names)
    print("mteb/scifact test exemplo:", ds[0])
except Exception as e:
    print("mteb/scifact erro:", e)