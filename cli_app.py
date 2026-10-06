from preprocessing import preprocess
from retrieval import tfidf_search
import pandas as pd

df = pd.read_csv("data/corpus_arguana_texto_plano.csv")
docs = df["Texto"].tolist()

query = input("Consulta: ")
query_pre = preprocess(query)
ranked_indices, scores = tfidf_search(query_pre, docs)

print("\nRanking de documentos:")
for i in ranked_indices[:10]:
    print(f"[{i}] Score: {scores[i]:.4f}")
    print(docs[i])
    print("=" * 50)