from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def tfidf_search(query, docs):
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(docs)
    query_vec = vect.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return np.argsort(scores)[::-1], scores

def bm25_search(query, docs, k1=1.5, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    query_terms = query.lower().split()
    doc_len = X.sum(axis=1)
    avg_len = doc_len.mean()
    scores = []

    for i in range(X.shape[0]):
        score = 0
        for term in query_terms:
            if term in terms:
                df = np.count_nonzero(X[:, vectorizer.vocabulary_[term]].toarray())
                idf = np.log((X.shape[0] - df + 0.5) / (df + 0.5) + 1)
                tf = X[i, vectorizer.vocabulary_[term]]
                denom = tf + k1 * (1 - b + b * doc_len[i, 0] / avg_len)
                score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return np.argsort(scores)[::-1], scores