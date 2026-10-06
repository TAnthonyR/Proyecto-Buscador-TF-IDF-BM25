import ir_datasets
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Función de preprocesamiento de documentos
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)

# Cargar el corpus
dataset = ir_datasets.load("beir/arguana")
docs_data = []

for i, doc in enumerate(dataset.docs_iter(), start=1):
    if doc.text and doc.text.strip():
        docs_data.append({
            'ID': i,
            'Doc_ID': doc.doc_id,
            'Texto_original': doc.text,
            'Texto_preprocesado': preprocess(doc.text)
        })

df = pd.DataFrame(docs_data)
df.to_csv("data/corpus_arguana_preprocesado.csv", index=False)