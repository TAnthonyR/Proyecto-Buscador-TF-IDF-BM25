from preprocessing import preprocess
from retrieval import tfidf_search
import pandas as pd

df = pd.read_csv("data/corpus_arguana_preprocesado.csv")
df = df.dropna(subset=["Texto_preprocesado", "Texto_original"]).reset_index(drop=True)
docs = df["Texto_preprocesado"].tolist()

query = input("Consulta: ")
query_pre = preprocess(query)
ranked_indices, scores = tfidf_search(query_pre, docs)

print("\nRanking de documentos:")
for i in ranked_indices[:10]:
    print(f"[{i}] Score: {scores[i]:.4f}")
    print(df["Texto_original"].iloc[i])
    print("=" * 50)
