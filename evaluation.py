import pandas as pd

def evaluate(qrels_path, retrieved_ids, query_id, top_k=10, debug=False):
    qrels = pd.read_csv(qrels_path, sep='\t', header=0)  # Usa encabezado original del archivo
    qrels["relevance"] = pd.to_numeric(qrels["relevance"], errors="coerce")
    qrels = qrels.dropna(subset=["relevance"])
    qrels["relevance"] = qrels["relevance"].astype(int)
    relevant_docs = set(qrels[(qrels["query_id"] == query_id) & (qrels["relevance"] > 0)]["doc_id"])

    if debug:
        print("¡DEBUG ACTIVADO!")
        print(f"Relevantes para query_id {query_id}: {relevant_docs}")
        print(f"Top recuperados: {retrieved_ids[:top_k]}")

    if not relevant_docs:
        return 0.0, 0.0, 0.0

    retrieved_at_k = retrieved_ids[:top_k]
    retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant_docs]

    if debug:
        print(f"Coincidencias relevantes encontradas: {retrieved_relevant}")

    precision = len(retrieved_relevant) / top_k
    recall = len(retrieved_relevant) / len(relevant_docs)

    if not retrieved_relevant:
        map_score = 0.0
    else:
        map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
            [i for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs])]) / len(relevant_docs)

    return precision, recall, map_score