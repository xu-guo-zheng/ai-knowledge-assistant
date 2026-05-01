import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data"


def load_documents() -> List[str]:
    docs = []

    if not os.path.exists(DATA_DIR):
        return docs

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = split_text(content)
            docs.extend(chunks)

    return docs


def split_text(text: str, chunk_size: int = 200) -> List[str]:
    text = text.replace("\n", " ").strip()
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def retrieve(query: str, top_k: int = 3) -> List[str]:
    documents = load_documents()

    if not documents:
        return []

    vectorizer = TfidfVectorizer(analyzer="char")
    vectors = vectorizer.fit_transform(documents + [query])

    doc_vectors = vectors[:-1]
    query_vector = vectors[-1]

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        if similarities[index] > 0:
            results.append(documents[index])

    return results