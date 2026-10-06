import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)

def preprocess_corpus(csv_path):
    df = pd.read_csv(csv_path)
    df["Texto_preprocesado"] = df["Texto"].apply(preprocess)
    return df

def preprocess_queries_tsv(tsv_path, output_path="queries_preprocessed.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")
    df["text_proc"] = df["text"].apply(preprocess)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Archivo guardado como {output_path}")
    return df
# Preprocesar archivo
# preprocess_queries_tsv("data/queries.tsv")

def preprocess_query(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)


