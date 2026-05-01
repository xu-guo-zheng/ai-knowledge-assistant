# AI Knowledge Assistant (RAG)

## 📌 项目简介
这是一个基于 **RAG（Retrieval-Augmented Generation）** 的 AI 知识库问答系统。

用户可以上传文档，系统会对文档进行向量化处理，并通过 FAISS 进行语义检索，再结合大模型生成回答，从而实现对私有知识库的智能问答。

---

## ⚙️ 技术栈
- Python
- FastAPI
- FAISS（向量数据库）
- SentenceTransformers（Embedding）
- RAG（检索增强生成）

---

## 🔥 核心功能
- 📄 文档上传与管理（支持动态更新知识库）
- 🔍 向量检索（基于 FAISS 实现语义搜索）
- 💬 多轮对话（Memory 机制）
- ⚡ 缓存机制（避免重复调用模型，提高响应速度）
- 🛡 fallback机制（模型失败时返回检索结果）
- 📝 问答日志记录

---

## 🧠 系统架构（RAG流程）

用户问题  
↓  
向量检索（FAISS）  
↓  
获取相关上下文（contexts）  
↓  
拼接 Prompt  
↓  
调用大模型生成答案  

---

## 🚀 启动方式

```bash
pip install -r requirements.txt
python -m uvicorn main:app
