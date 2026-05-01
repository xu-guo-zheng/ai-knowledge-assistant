import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from datetime import datetime
import json

from rag import retrieve,init_vector_store,rebuild_vector_store


load_dotenv()

app = FastAPI(title="AI Knowledge Assistant")
# 简单内存版对话历史
# key：session_id，代表一次会话
# value：该会话的消息列表
# 注意：这是内存存储，服务重启后会丢失
chat_memory = {}
# 简单内存缓存（模拟redis）
qa_cache = {}
@app.on_event("startup")
def startup_event():
    """
    服务启动时初始化向量库。

    逻辑：
    - 有本地 FAISS 索引：直接加载
    - 没有索引：自动构建
    """
    init_vector_store()

@app.post("/rebuild")
def rebuild():
    """
    手动重建 FAISS 向量索引。

    使用场景：
    - 修改了 data/company.txt
    - 新增了 txt 文件
    - 删除了知识库文件

    为什么需要这个接口：
    - 因为 FAISS 索引是保存到磁盘的
    - 文档变了以后，需要重新生成索引
    """
    rebuild_vector_store()

    return {
        "message": "知识库索引重建完成"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传 txt 文档到知识库目录。

    实现逻辑：
    1. 检查上传文件是否是 txt
    2. 保存到 data 目录
    3. 返回保存结果

    注意：
    - 上传后还需要调用 /rebuild
    - 因为 FAISS 索引不会自动知道文件变了
    """

    # 只允许上传 txt，避免后面解析复杂格式
    if not file.filename.endswith(".txt"):
        return {
            "message": "目前只支持上传 .txt 文件"
        }

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # 拼接保存路径
    save_path = os.path.join("data", file.filename)

    # 读取上传文件内容
    content = await file.read()

    # 保存文件
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "message": "文件上传成功",
        "filename": file.filename,
        "next_step": "请调用 /rebuild 重建知识库索引"
    }

@app.get("/documents")
def list_documents():
    """
    查看当前知识库中的 txt 文档。

    作用：
    - 方便确认 data 目录里有哪些知识库文件
    - 上传后可以检查文件是否保存成功
    """
    os.makedirs("data", exist_ok=True)

    files = []

    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            file_path = os.path.join("data", filename)
            files.append({
                "filename": filename,
                "size": os.path.getsize(file_path)
            })

    return {
        "documents": files
    }

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    删除知识库中的指定 txt 文档。

    作用：
    - 如果上传错文件，可以删除
    - 删除后需要重新调用 /rebuild，让 FAISS 索引同步更新
    """
    if not filename.endswith(".txt"):
        return {
            "message": "只能删除 .txt 文件"
        }

    file_path = os.path.join("data", filename)

    if not os.path.exists(file_path):
        return {
            "message": "文件不存在"
        }

    os.remove(file_path)

    return {
        "message": "文件删除成功",
        "filename": filename,
        "next_step": "请调用 /rebuild 重建知识库索引"
    }

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL_ID")


class ChatRequest(BaseModel):
    message: str


class AskRequest(BaseModel):
    question: str

class MemoryChatRequest(BaseModel):
    """
    多轮对话请求体。

    session_id:
    - 用来区分不同用户/不同会话
    - 比如 user001、test_session

    message:
    - 用户当前输入的问题
    """
    session_id: str
    message: str

@app.get("/")
def root():
    return {"message": "AI Knowledge Assistant is running"}


def call_llm(messages, temperature: float = 0.7):
    url = f"{BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    result = response.json()

    if "error" in result:
        return f"模型调用失败：{result['error']}"

    return result["choices"][0]["message"]["content"]


@app.post("/chat")
def chat(req: ChatRequest):
    answer = call_llm([
        {"role": "system", "content": "你是一个专业、简洁的AI助手。"},
        {"role": "user", "content": req.message}
    ])

    return {"answer": answer}


@app.post("/ask")
def ask(req: AskRequest):
    """
    RAG 知识库问答接口。

    执行流程：
    1. 根据用户问题检索知识库
    2. 如果没有检索结果，直接返回“未找到”
    3. 如果有检索结果，把内容交给大模型生成答案
    4. 如果模型调用失败，启用 fallback，直接返回检索内容
    5. 保存问答日志
    """
    cache_key = req.question.strip()

    # 命中缓存
    if cache_key in qa_cache:
        return {
            **qa_cache[cache_key],
            "cached": True
        }

    # 第一步：从 FAISS 知识库中检索相关文本片段
    contexts = retrieve(req.question, top_k=3)

    # 第二步：如果没有检索到内容，就不调用大模型，避免模型乱编
    if not contexts:
        answer = "知识库中没有检索到相关内容。"
        save_qa_log(req.question, answer, contexts)

        return {
            "answer": answer,
            "contexts": contexts
        }

    # 第三步：把检索到的内容拼成上下文
    # 给每段 context 编号（方便引用）
    context_text = ""
    for i, ctx in enumerate(contexts):
        context_text += f"[{i+1}] {ctx}\n\n"
    
    # 第四步：构造提示词，要求模型严格基于知识库回答
    prompt = f"""
        你是一个知识库问答助手。请严格根据下面的知识库内容回答用户问题。
        如果知识库内容不足以回答，请直接说“知识库中没有足够信息”。
        要求：
        1. 回答要简洁清晰
        2. 可以用分点形式
        3. 不要编造知识库中没有的信息
        4. 如果信息不足，请明确说明

        【知识库内容】
        {context_text}

        【用户问题】
        {req.question}
        请输出结构化回答：
        """
    # 第五步：调用大模型生成答案
    answer = call_llm([
        {"role": "system", "content": "你是一个严谨的知识库问答助手。"},
        {"role": "user", "content": prompt}
    ], temperature=0.2)

    # 第六步：如果模型调用失败，启用 fallback，直接返回检索内容
    # 🔥 fallback逻辑：模型失败时兜底
    # 判断方式：你当前的 call_llm 返回的是字符串
    # 如果包含“模型调用失败”，说明接口挂了
    if answer.startswith("模型调用失败"):

        # 👉 fallback：不用模型生成，直接用检索内容拼一个答案
        fallback_answer = "当前模型不可用，基于知识库内容：\n\n"

        for i, ctx in enumerate(contexts):
            fallback_answer += f"{i+1}. {ctx}\n"
        # 保存 fallback 日志，方便后续排查模型失败原因
        save_qa_log(req.question, fallback_answer, contexts)
        
        return {
            "answer": fallback_answer,
            "contexts": contexts,
            "note": "这是模型调用失败时的 fallback 答案，直接返回检索内容。"

        }
    # 第七步：正常保存问答日志并返回模型答案
    save_qa_log(req.question, answer, contexts)

    return {
        "answer": answer,
        "sources": [
            {
                "id": i + 1,
                "content": ctx[:100]  # 只展示前100字
            }
            for i, ctx in enumerate(contexts)
        ]
    }

    # 存入缓存
    qa_cache[cache_key] = result

    return {
        **result,
        "cached": False
    }


def save_qa_log(question: str, answer: str, contexts: list):
    """
    保存问答日志。

    实现逻辑：
    1. 创建 logs 目录
    2. 把每次问答保存成一行 JSON
    3. 使用 jsonl 格式，方便后续一行一行读取

    为什么要这样做：
    - 方便排查模型回答问题
    - 后续可以做历史记录页面
    - 项目更像真实 AI 应用
    """
    os.makedirs("logs", exist_ok=True)

    log_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "contexts": contexts
    }

    with open("logs/qa_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


@app.get("/logs")
def get_logs(limit: int = 20):
    """
    查看最近的问答日志。

    参数：
    - limit：最多返回多少条

    作用：
    - 查看历史提问
    - 检查模型回答和检索内容
    """
    log_path = "logs/qa_logs.jsonl"

    if not os.path.exists(log_path):
        return {
            "logs": []
        }

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    recent_lines = lines[-limit:]

    logs = [json.loads(line) for line in recent_lines]

    return {
        "logs": logs
    }

@app.post("/chat-memory")
def chat_memory_api(req: MemoryChatRequest):
    """
    多轮对话接口。

    实现逻辑：
    1. 根据 session_id 获取历史对话
    2. 把历史对话 + 当前问题一起发给模型
    3. 保存本轮用户输入和模型回答
    4. 只保留最近 10 条消息，避免上下文太长
    """

    # 如果这个 session_id 还没有历史记录，就初始化一个空列表
    if req.session_id not in chat_memory:
        chat_memory[req.session_id] = []

    history = chat_memory[req.session_id]

    # 构造发送给大模型的 messages
    messages = [
        {
            "role": "system",
            "content": "你是一个专业、简洁、有上下文记忆能力的AI助手。"
        }
    ]

    # 加入历史对话
    messages.extend(history)

    # 加入当前用户问题
    messages.append({
        "role": "user",
        "content": req.message
    })

    # 调用大模型
    answer = call_llm(messages, temperature=0.7)

    # 保存当前用户问题
    history.append({
        "role": "user",
        "content": req.message
    })
    if answer.startswith("模型调用失败"):
        return {
            "answer": answer,
            "session_id": req.session_id,
            "history_count": len(history),
            "note": "模型调用失败，本轮未写入对话历史。"
        }
    # 保存模型回答
    history.append({
        "role": "assistant",
        "content": answer
    })

    # 只保留最近 10 条消息，防止上下文无限增长
    chat_memory[req.session_id] = history[-10:]

    return {
        "answer": answer,
        "session_id": req.session_id,
        "history_count": len(chat_memory[req.session_id])
    }