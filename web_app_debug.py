from flask import Flask, request, render_template_string
from preprocessing import preprocess_query
from retrieval import tfidf_search, bm25_search
import pandas as pd
import importlib
import evaluation
importlib.reload(evaluation)


app = Flask(__name__)
df = pd.read_csv("data/corpus_arguana_preprocesado.csv")
df = df.dropna(subset=["Texto_preprocesado", "Texto_original"]).reset_index(drop=True)
docs = df["Texto_preprocesado"].tolist()
doc_ids = df["Doc_ID"].tolist()

df_qrels = pd.read_csv("data/qrels.tsv", sep="\t")
query_ids = sorted(df_qrels["query_id"].dropna().unique())

HTML = """
<form method="post">
  <label>Consulta:</label><br>
  <input name="query" style="width: 300px;" value="{{ query }}">
  <br><br>
  <label>Método de búsqueda:</label><br>
  <select name="method">
    <option value="tfidf" {% if method == 'tfidf' %}selected{% endif %}>TF-IDF</option>
    <option value="bm25" {% if method == 'bm25' %}selected{% endif %}>BM25</option>
  </select>
  <br><br>
  <label>Seleccionar Query ID evaluado:</label><br>
  <select name="query_id">
    {% for qid in query_ids %}
      <option value="{{ qid }}" {% if qid == query_id %}selected{% endif %}>{{ qid }}</option>
    {% endfor %}
  </select>
  <br><br>
  <br><br>
<label>Seleccionar Query ID evaluado:</label><br>
<select name="query_id">
  {% for qid in query_ids %}
    <option value="{{ qid }}" {% if qid == query_id %}selected{% endif %}>{{ qid }}</option>
  {% endfor %}
</select>
  <input type="submit" value="Buscar">
</form>
{% if query %}
  <p><strong>Consulta ingresada:</strong> {{ query }}</p>
  <p><strong>Método:</strong> {{ method|upper }}</p>
  <p><strong>Query ID:</strong> {{ query_id }}</p>
  <p><strong>Precisión:</strong> {{ precision }}</p>
  <p><strong>Recall:</strong> {{ recall }}</p>
  <p><strong>MAP:</strong> {{ map_score }}</p>
{% endif %}
{% for i, score, doc in results %}
  <p><b>{{ i }} - {{ score }}</b></p>
  <p>{{ doc }}</p>
  <hr>
{% endfor %}
"""

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    method = "tfidf"
    query_id = query_ids[0]
    precision = recall = map_score = 0.0

    if request.method == "POST":
        query = request.form["query"]
        method = request.form["method"]
        query_id = request.form["query_id"]
        query_pre = preprocess_query(query)

        if method == "bm25":
            ranked_indices, scores = bm25_search(query_pre, docs)
        else:
            ranked_indices, scores = tfidf_search(query_pre, docs)

        ranked_doc_ids = [doc_ids[i] for i in ranked_indices]

        try:
           print("LLAMADO A EVALUATE")
           precision, recall, map_score = evaluation.evaluate("data/qrels.tsv", ranked_doc_ids, query_id=query_id, debug=True)
           print("EVALUATE COMPLETADO")
        except Exception as e:
            print("ERROR en evaluate:", e)
            precision, recall, map_score = 0.0, 0.0, 0.0

        results = [(i, f"{scores[i]:.3f}", df['Texto_original'][i]) for i in ranked_indices[:10]]

    return render_template_string(
        HTML,
        results=results,
        query=query,
        method=method,
        precision=f"{precision:.3f}",
        recall=f"{recall:.3f}",
        map_score=f"{map_score:.3f}",
        query_id=query_id,
        query_ids=query_ids
    )

if __name__ == "__main__":
    app.run(debug=True)