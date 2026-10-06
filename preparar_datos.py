"""Reconstruye corpus, consultas y relevancias de ArguAna desde la raíz."""
from pathlib import Path
import os
import runpy
import nltk

def main():
    os.chdir(Path(__file__).resolve().parent)
    Path("data").mkdir(exist_ok=True)
    for resource in ("punkt", "punkt_tab", "stopwords"):
        if not nltk.download(resource):
            raise RuntimeError(f"No se pudo descargar {resource}")
    runpy.run_path("extract_corpus.py", run_name="__main__")
    runpy.run_path("generar_qrels_y_queries.py", run_name="__main__")
    from preprocessing import preprocess_queries_tsv
    preprocess_queries_tsv("data/queries.tsv", "data/queries_preprocessed.tsv")
    print("Datos preparados. Ejecuta python web_app_auto_queryid_preprocessed.py")

if __name__ == "__main__":
    main()
