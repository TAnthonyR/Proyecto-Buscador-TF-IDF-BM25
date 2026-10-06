"""Ejecuta los buscadores originales sobre un corpus pequeno de demostracion."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retrieval import tfidf_search, bm25_search

docs = [
    'solar energy renewable electricity clean solar power',
    'wind energy renewable electricity wind turbines',
    'coal electricity fossil fuel pollution',
    'solar panels generate solar electricity',
    'libraries books reading education',
    'renewable energy solar wind clean electricity research',
]
query = 'solar renewable energy'
_, tfidf = tfidf_search(query, docs)
_, bm25 = bm25_search(query, docs)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
for ax, scores, title in zip(axes, [tfidf, bm25], ['TF-IDF + coseno', 'BM25']):
    ax.barh([f'Documento {i+1}' for i in range(len(docs))], scores, color='#2678b8')
    ax.set_xlabel('Puntuacion (escala propia de cada algoritmo)')
    ax.set_title(title)
    ax.invert_yaxis()
fig.suptitle(f'Demostracion ejecutada · consulta: {query}', fontsize=13)
fig.savefig('preview-busqueda.png', dpi=160)
print('Figura guardada: preview-busqueda.png')
