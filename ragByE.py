import os
from typing import List
import numpy as np

# 使用 sentence-transformers 做语义向量
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data"

# 加载 embedding 模型（只会加载一次）
# 这是一个轻量模型，适合本地运行
# model = SentenceTransformer("all-MiniLM-L6-v2")
# 使用本地中文 embedding 模型，避免每次联网下载
model = SentenceTransformer(r"D:\gitDemo\ai-knowledge-assistant\model\text2vec-base-chinese-sentence")

def load_documents() -> List[str]:
    """
    读取 data 目录下的所有 txt 文件，并切分成小块
    """
    docs = []

    if not os.path.exists(DATA_DIR):
        return docs

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # 切分文本（避免太长）
            chunks = split_text(content)
            docs.extend(chunks)

    return docs


def split_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    把长文本切分成多个小段（chunk）
    原因：embedding对短文本效果更好
    """
    text = text.replace("\n", " ").strip()
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    核心检索函数（RAG中的 Retrieval）
    """

    documents = load_documents()

    if not documents:
        return []

    # 🔥 第一步：把文档转成向量
    doc_embeddings = model.encode(documents)

    # 🔥 第二步：把用户问题转成向量
    query_embedding = model.encode([query])

    # 🔥 第三步：计算相似度（余弦相似度）
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # 🔥 第四步：取最相似的 top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # 过滤太低的匹配
            results.append(documents[idx])

    return results