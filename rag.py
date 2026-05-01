import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 知识库目录
DATA_DIR = "data"

# FAISS 索引保存目录
INDEX_DIR = "vector_store"

# FAISS 索引文件
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

# 文本片段保存文件
DOCS_PATH = os.path.join(INDEX_DIR, "documents.pkl")


# 加载本地 embedding 模型
# 作用：把文本转成向量，用于语义检索
model = SentenceTransformer(
    r"D:\gitDemo\ai-knowledge-assistant\model\text2vec-base-chinese-sentence"
)


# 内存缓存：保存文档文本
documents_cache: List[str] = []

# 内存缓存：保存 FAISS 索引对象
faiss_index = None


def split_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    把长文本切分成多个小片段。

    为什么要切分：
    - 长文档直接做 embedding 效果不好
    - RAG 更适合检索局部相关片段
    - 可以减少传给大模型的 token
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
    读取 data 目录下所有 txt 文件，并切分成片段。
    """
    docs = []

    if not os.path.exists(DATA_DIR):
        return docs

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            docs.extend(split_text(content))

    return docs


def build_faiss_index():
    """
    构建 FAISS 向量索引，并保存到磁盘。

    做了什么：
    1. 读取文档
    2. 切分文档
    3. 生成 embedding 向量
    4. 用 FAISS 建立索引
    5. 保存 index.faiss 和 documents.pkl

    为什么要保存：
    - 避免每次启动都重新生成文档向量
    - 提升启动速度
    - 让知识库索引持久化
    """
    global documents_cache, faiss_index

    documents_cache = load_documents()

    if not documents_cache:
        print("知识库为空，无法构建 FAISS 索引")
        return

    # 生成文档向量
    embeddings = model.encode(documents_cache)

    # FAISS 需要 float32 类型
    embeddings = np.array(embeddings).astype("float32")

    # 获取向量维度，比如 384 或 768
    dimension = embeddings.shape[1]

    # IndexFlatL2 是最基础的 FAISS 索引
    # L2 表示欧式距离，距离越小越相似
    faiss_index = faiss.IndexFlatL2(dimension)

    # 把文档向量加入索引
    faiss_index.add(embeddings)

    # 创建保存目录
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 保存 FAISS 索引
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    # 保存文本片段，方便通过检索结果 id 找回原文
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents_cache, f)

    print(f"FAISS 索引构建完成，共保存 {len(documents_cache)} 个文本片段")


def load_faiss_index():
    """
    从磁盘加载 FAISS 索引和文本片段。

    为什么要加载：
    - 服务重启后不用重新计算所有文档向量
    """
    global documents_cache, faiss_index

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("未找到本地 FAISS 索引，需要先构建")
        return False

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        documents_cache = pickle.load(f)

    print(f"FAISS 索引加载完成，共加载 {len(documents_cache)} 个文本片段")
    return True


def init_vector_store():
    """
    初始化向量库。

    执行逻辑：
    - 如果本地已有 FAISS 索引，就直接加载
    - 如果没有，就读取文档并构建索引
    """
    loaded = load_faiss_index()

    if not loaded:
        build_faiss_index()


def rebuild_vector_store():
    """
    强制重建向量库。

    使用场景：
    - data 目录里的文档发生变化
    - 新增/删除文档后，需要重新生成索引
    """
    build_faiss_index()


def retrieve(query: str, top_k: int = 3) -> List[str]:
    """
    根据用户问题，从 FAISS 索引中检索最相关的文本片段。
    """
    if faiss_index is None or not documents_cache:
        return []

    # 把用户问题转成向量
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # FAISS search 返回：
    # distances: 距离，越小越相似
    # indices: 文档片段编号
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        if idx == -1:
            continue

        results.append(documents_cache[idx])

    return results