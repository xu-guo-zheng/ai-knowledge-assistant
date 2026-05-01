
# 向量存在内存中，不使用外部数据库
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 知识库文本所在目录
DATA_DIR = "data"

# 加载本地 Embedding 模型
# 作用：把文本转成向量，用于语义检索
# 注意：这里用你的本地模型路径，避免联网下载
model = SentenceTransformer(
    r"D:\gitDemo\ai-knowledge-assistant\model\text2vec-base-chinese-sentence"
)

# 全局缓存：保存切分后的文本片段
documents_cache: List[str] = []

# 全局缓存：保存文本片段对应的向量
embeddings_cache = None


def split_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    把长文本切分成多个小片段。

    为什么要切分：
    1. 文本太长会影响检索准确性
    2. RAG 通常检索的是“相关片段”，不是整篇文档
    3. 后面把片段传给大模型，也能减少 token 消耗
    """
    text = text.replace("\n", " ").strip()
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def load_documents() -> List[str]:
    """
    读取 data 目录下所有 txt 文件，并切分成文本片段。
    """
    docs = []

    if not os.path.exists(DATA_DIR):
        return docs

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = split_text(content)
            docs.extend(chunks)

    return docs


def build_index():
    """
    构建内存向量索引。

    什么时候执行：
    - 服务启动时执行一次

    做了什么：
    1. 读取知识库文档
    2. 切分成片段
    3. 把每个片段转成 embedding 向量
    4. 存到全局缓存里

    为什么这么做：
    - 避免每次用户提问时重复计算文档向量
    - 提升 /ask 接口响应速度
    """
    global documents_cache, embeddings_cache

    documents_cache = load_documents()

    if not documents_cache:
        embeddings_cache = None
        print("知识库为空，没有构建向量索引")
        return

    embeddings_cache = model.encode(documents_cache)

    print(f"向量索引构建完成，共加载 {len(documents_cache)} 个文本片段")


def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    根据用户问题，从缓存的知识库向量中检索最相关的文本片段。

    参数：
    - query：用户问题
    - top_k：最多返回几个相关片段

    返回：
    - 最相关的文本片段列表
    """
    if not documents_cache or embeddings_cache is None:
        return []

    # 把用户问题转成向量
    query_embedding = model.encode([query])

    # 计算用户问题向量与每个文档片段向量的相似度
    similarities = cosine_similarity(query_embedding, embeddings_cache)[0]

    # 按相似度从高到低排序，取前 top_k 个
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        score = similarities[index]

        # 相似度阈值：太低说明不相关，避免硬塞无关内容给大模型
        if score > 0.3:
            results.append(documents_cache[index])

    return results