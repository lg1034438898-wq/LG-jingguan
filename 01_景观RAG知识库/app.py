# -*- coding: utf-8 -*-
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import mysql.connector
import os
import requests
import jieba
from rank_bm25 import BM25Okapi
import time
import json
import concurrent.futures

# ==================== Rerank 依赖 ====================
try:
    from FlagEmbedding import FlagReranker
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="景观 AI 知识库",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 🌟 全局园林主题 CSS ====================
GLOBAL_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        color: #2C3E2D;
    }
    .stApp {
        background: linear-gradient(180deg, #F7F9F5 0%, #EFF4EB 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stHeader"] { background: transparent; height: auto; }
    [data-testid="stHeader"] > div { background: transparent; }

    [data-testid="collapsedControl"] {
        background: #7D9B78 !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3) !important;
        padding: 6px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: #5C7D55 !important;
        transform: scale(1.1) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    [data-testid="stStatusWidget"] {visibility: hidden;}

    .hero-title {
        font-family: 'Noto Serif SC', 'Songti SC', serif;
        font-size: 42px; font-weight: 700; color: #2D5016;
        letter-spacing: 2px; margin-bottom: 6px; line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 15px; color: #6B7B6A; letter-spacing: 1px;
        margin-bottom: 28px; padding-bottom: 20px;
        border-bottom: 1px solid #D8E2D3;
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF; border-radius: 16px;
        padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(45, 80, 22, 0.06);
        border: 1px solid #E5EDE1;
    }

    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important; border-radius: 14px !important;
        border: 1.5px solid #C6D6BC !important;
        font-size: 15px !important; padding: 14px 18px !important;
        box-shadow: 0 4px 16px rgba(45, 80, 22, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #7D9B78 !important;
        box-shadow: 0 4px 20px rgba(125, 155, 120, 0.25) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F0F5EC 0%, #E5EDE1 100%);
        border-right: 1px solid #D8E2D3;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 20px; }

    [data-testid="stSidebar"] .new-chat-btn button {
        background: linear-gradient(135deg, #7D9B78 0%, #5C7D55 100%) !important;
        color: #FFFFFF !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        font-size: 15px !important; padding: 12px 0 !important;
        box-shadow: 0 4px 12px rgba(93, 125, 85, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stSidebar"] .new-chat-btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(93, 125, 85, 0.45) !important;
    }

    [data-testid="stSidebar"] .history-btn button {
        background: transparent !important;
        color: #4A5D45 !important; border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 8px !important; text-align: left !important;
        padding: 8px 12px !important; font-size: 13px !important;
        font-weight: 400 !important; transition: all 0.2s ease !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .history-btn button:hover {
        background: #E5EDE1 !important;
        border-left-color: #7D9B78 !important;
    }

    .stButton button {
        background: #FFFFFF; color: #2D5016;
        border: 1.5px solid #C6D6BC; border-radius: 10px;
        font-weight: 500; transition: all 0.25s ease;
        padding: 8px 18px;
    }
    .stButton button:hover {
        background: #7D9B78; color: #FFFFFF;
        border-color: #7D9B78; transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(125, 155, 120, 0.35);
    }

    details {
        background: #FAFCF8; border-radius: 12px;
        border: 1px solid #E5EDE1; padding: 8px 14px; margin-top: 12px;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #F0F5EC; }
    ::-webkit-scrollbar-thumb { background: #B7C9AE; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #7D9B78; }

    .section-label {
        font-size: 12px; color: #7D9B78; letter-spacing: 1px;
        font-weight: 600; margin: 6px 0 10px 4px;
        text-transform: uppercase;
    }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ==================== 顶部 Hero ====================
st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div class="hero-title">🌿 景观 AI 知识库</div>
        <div class="hero-subtitle">LANDSCAPE AI KNOWLEDGE SYSTEM · 基于山水比德公众号与景观图集构建</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================== 🌟 植物生长加载动画 ====================
class PlantGrowthSpinner:
    def __init__(self, text):
        self.placeholder = st.empty()
        self._render("🌱", text, "正在生长...")

    def _render(self, emoji, text, caption=""):
        html_code = f"""
        <div style="display: flex; align-items: center; gap: 14px; color: #2D5016;
                    padding: 14px 20px; border-radius: 14px;
                    background: linear-gradient(90deg, #EFF4EB 0%, #E5EDE1 100%);
                    border-left: 4px solid #7D9B78;
                    box-shadow: 0 2px 8px rgba(45, 80, 22, 0.06);">
            <style>
                @keyframes grow {{
                    0% {{ transform: scale(0.5) translateY(5px); opacity: 0.5; }}
                    50% {{ transform: scale(1.15) translateY(0); opacity: 1; }}
                    100% {{ transform: scale(1) translateY(0); opacity: 1; }}
                }}
                .growing-plant {{ display: inline-block; font-size: 26px;
                    animation: grow 1.5s ease-in-out infinite alternate; }}
            </style>
            <span class="growing-plant">{emoji}</span>
            <div>
                <div style="font-size: 15px; font-weight: 600;">{text}</div>
                <div style="font-size: 12px; color: #7D9B78; margin-top: 2px;">{caption}</div>
            </div>
        </div>
        """
        self.placeholder.markdown(html_code, unsafe_allow_html=True)

    def update(self, new_text, emoji="🌿"):
        self._render(emoji, new_text)

    def empty(self):
        self.placeholder.empty()

# ==================== 🌟 会话管理初始化 ====================
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def save_current_session():
    if not st.session_state.current_session_id:
        return
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.current_session_id:
            s["messages"] = st.session_state.messages[:]
            return

def start_new_session():
    save_current_session()
    st.session_state.messages = []
    st.session_state.current_session_id = None

def switch_session(session_id):
    save_current_session()
    for s in st.session_state.sessions:
        if s["id"] == session_id:
            st.session_state.current_session_id = session_id
            st.session_state.messages = s["messages"][:]
            return

# ==================== 🌟 侧边栏 ====================
with st.sidebar:
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  开启新对话", use_container_width=True, key="new_chat_btn"):
        start_new_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">最近对话</div>', unsafe_allow_html=True)

    if not st.session_state.sessions:
        st.caption("暂无历史对话")
    else:
        for s in reversed(st.session_state.sessions[-10:]):
            is_active = (s["id"] == st.session_state.current_session_id)
            prefix = "● " if is_active else "○ "
            display_title = s["title"][:20] + ("..." if len(s["title"]) > 20 else "")

            st.markdown('<div class="history-btn">', unsafe_allow_html=True)
            if st.button(f"{prefix}{display_title}",
                         key=f"switch_{s['id']}",
                         use_container_width=True):
                switch_session(s["id"])
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 🌟 检索设置：Rerank 默认关闭
    with st.expander("⚙️ 检索设置", expanded=False):
        use_rerank = st.checkbox(
            "开启 Rerank 精排",
            value=False,   # 🌟 关键修改：默认关闭，避免算力不足
            help="Rerank 能提升检索精度，但会占用大量算力。若本机算力不足，建议关闭。"
        )
        st.caption(f"Rerank 模型：{'✅ 就绪' if RERANK_AVAILABLE else '❌ 未安装'}")
        st.caption("💡 若开启后生成超时，说明本机算力不足。")

if "use_rerank" not in locals():
    use_rerank = False

# ==================== 全局配置 ====================
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD",   # ← 改成这个
    "database": "jingguan_rag"
}
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "landscape_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5vl:7b"   # 如果一直超时，可换成 "deepseek-r1:1.5b"

# ==================== 资源初始化 ====================
@st.cache_resource
def init_connections():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, content FROM chunks WHERE status='active'")
    all_rows = cursor.fetchall()
    if not all_rows:
        return collection, conn, [], None, None

    all_ids = [row[0] for row in all_rows]
    all_texts = [row[1] for row in all_rows]
    tokenized_corpus = [list(jieba.cut(text)) for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    reranker = None
    if RERANK_AVAILABLE:
        try:
            reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        except Exception as e:
            print(f"⚠️ Rerank 模型加载失败: {e}")

    return collection, conn, all_ids, bm25, reranker

collection, conn, bm25_ids, bm25, reranker = init_connections()
cursor = conn.cursor()

# ==================== 聊天历史渲染 ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎨" if msg["role"] == "user" else "🌿"):
        st.markdown(msg["content"])

# ==================== 聊天交互主逻辑 ====================
if prompt := st.chat_input("🌱 请输入景观相关问题，例如：适合重庆种植的中阳性树种有哪些？"):
    if st.session_state.current_session_id is None:
        new_id = f"session_{int(time.time() * 1000)}"
        title = prompt[:20] + ("..." if len(prompt) > 20 else "")
        st.session_state.sessions.append({
            "id": new_id,
            "title": title,
            "messages": []
        })
        st.session_state.current_session_id = new_id

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎨"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌿"):
        start_time = time.time()

        # ========== 查询改写 ==========
        spinner = PlantGrowthSpinner("正在理解并优化您的提问...")

        rewrite_prompt = f"""你是一个景观设计领域的专家。请将用户的口语化问题改写为更适合知识库检索的专业景观术语。
要求：
1. 只输出改写后的句子，不要有任何解释、前缀、后缀或客套话。
2. 保留原问题的核心意图（如：植物种类、施工规范、设计原则）。
原问题：{prompt}"""

        try:
            rewrite_resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": rewrite_prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "num_ctx": 2048}
                },
                timeout=60
            )
            rewritten_query = rewrite_resp.json().get("response", "").strip()
            if not rewritten_query:
                rewritten_query = prompt
        except Exception:
            rewritten_query = prompt

        spinner.update("正在检索知识库中...", emoji="🌿")

        # ========== 混合检索 ==========
        vector_results = collection.query(
            query_texts=[rewritten_query], n_results=20, where={"status": "active"}
        )
        vector_ids = vector_results["ids"][0]

        if bm25 is not None:
            tokenized_query = list(jieba.cut(rewritten_query))
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_top_indices = bm25_scores.argsort()[-20:][::-1]
            bm25_top_ids = [bm25_ids[i] for i in bm25_top_indices]
        else:
            bm25_top_ids = []

        rrf_scores = {}
        k = 60
        for rank, doc_id in enumerate(vector_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(bm25_top_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        candidate_ids = [doc_id for doc_id, score in sorted_ids]

        if candidate_ids:
            placeholders = ",".join(["%s"] * len(candidate_ids))
            sql = f"SELECT chunk_id, content, page_number, chunk_type, image_path, doc_name FROM chunks WHERE status='active' AND chunk_id IN ({placeholders})"
            cursor.execute(sql, candidate_ids)
            rows = cursor.fetchall()
            candidate_map = {r[0]: r for r in rows}
        else:
            candidate_map = {}

        # ========== Rerank 精排（带超时降级） ==========
        if use_rerank and reranker is not None and candidate_map:
            pairs = [[rewritten_query, candidate_map[cid][1]] for cid in candidate_ids if cid in candidate_map]
            if pairs:
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(reranker.compute_score, pairs, normalize=True)
                        scores = future.result(timeout=5)
                    scored_candidates = [(cid, scores[i]) for i, cid in enumerate(candidate_ids) if cid in candidate_map]
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    chunk_ids = [item[0] for item in scored_candidates[:3]]
                except concurrent.futures.TimeoutError:
                    st.warning("⚠️ Rerank 超时，已自动降级为 RRF 排序。")
                    chunk_ids = candidate_ids[:3]
                except Exception:
                    chunk_ids = candidate_ids[:3]
            else:
                chunk_ids = candidate_ids[:3]
        else:
            chunk_ids = candidate_ids[:3]

        row_map = {cid: candidate_map[cid] for cid in chunk_ids if cid in candidate_map}

        # ========== 生成回答 ==========
        spinner.update("正在生成回答...", emoji="🌳")

        context_parts, image_paths_to_show, sources_info, retrieved_types = [], [], [], set()
        for i, cid in enumerate(chunk_ids):
            if cid in row_map:
                _, content, page, ctype, img_path, doc_name = row_map[cid]
                tag = "📷 图片描述" if ctype == "image_description" else ("📊 表格" if ctype == "table" else "📄 文本")
                context_parts.append(f"【资料{i + 1} | 来源：{doc_name} | 第{page}页 | {tag}】\n{content}")
                sources_info.append(f"`{doc_name}` · 第{page}页 · {tag}")
                retrieved_types.add(ctype)
                if ctype == "image_description" and img_path:
                    image_paths_to_show.append(img_path)

        context_text = "\n\n".join(context_parts)
        # 🌟 关键修改：上下文从 1500 字降至 800 字
        if len(context_text) > 800:
            context_text = context_text[:800] + "\n\n...(上下文过长，已截断)"

        if not context_text:
            st.warning("⚠️ 知识库中暂无相关数据，请尝试换一种问法。")
            spinner.empty()
        else:
            final_prompt = f"""你是专业的景观设计AI助手。请严格基于以下参考材料回答用户问题。
要求：
1. 只基于参考材料中的信息回答，绝对不要编造。
2. 材料中可能包含OCR识别导致的乱码，请忽略乱码，只提取有效信息。
3. 在回答中必须标注信息来源（如"根据《XXX》第X页"）。
4. 如果材料无法回答该问题，请直接说"知识库中暂无相关数据"。

参考材料：
{context_text}

用户问题：{prompt}
回答："""

            try:
                # 🌟 关键修改：加上 num_ctx 和 num_predict，超时放宽到 180 秒
                final_resp = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": LLM_MODEL,
                        "prompt": final_prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 400,   # 最多生成约400个token
                            "num_ctx": 2048       # 限制上下文窗口
                        }
                    },
                    timeout=180
                )
                final_answer = final_resp.json().get("response", "").strip()
                if not final_answer:
                    final_answer = f"⚠️ 大模型未返回有效内容。以下是原始资料：\n\n{context_text[:800]}"
            except Exception as e:
                final_answer = (f"⚠️ 本地算力不足，生成超时（已等待180秒）。建议在 `app.py` 中把 "
                                f"`LLM_MODEL` 换成 `deepseek-r1:1.5b` 或 `qwen2:0.5b`。\n\n"
                                f"以下是系统检索到的资料：\n\n{context_text[:800]}")

            spinner.empty()
            st.markdown(final_answer)

            # ========== 埋点 ==========
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            retrieved_types_str = ",".join(retrieved_types)

            insert_log_sql = """
                INSERT INTO retrieval_logs 
                (user_query, rewritten_query, retrieved_chunk_ids, retrieved_types, response_time_ms, is_adopted) 
                VALUES (%s, %s, %s, %s, %s, 0)
            """
            log_id = None
            try:
                cursor.execute(insert_log_sql, (
                    prompt, rewritten_query, json.dumps(chunk_ids),
                    retrieved_types_str, response_time_ms
                ))
                conn.commit()
                log_id = cursor.lastrowid
            except Exception as e:
                st.error(f"埋点写入失败: {e}")

            if log_id:
                st.caption(f"⏱️ 耗时 {response_time_ms} ms · Rerank {'✅' if (use_rerank and reranker) else '❌'}")
                col1, col2, _ = st.columns([1, 1, 6])
                with col1:
                    if st.button("✅ 采纳", key=f"adopt_{log_id}"):
                        cursor.execute("UPDATE retrieval_logs SET is_adopted=1 WHERE log_id=%s", (log_id,))
                        conn.commit()
                        st.toast("已记录您的采纳！")
                with col2:
                    if st.button("❌ 驳回", key=f"reject_{log_id}"):
                        cursor.execute("UPDATE retrieval_logs SET is_adopted=-1 WHERE log_id=%s", (log_id,))
                        conn.commit()
                        st.toast("已记录您的反馈！")

            with st.expander("🔍 查看检索到的原始资料"):
                for s in sources_info:
                    st.markdown(f"- {s}")
                st.markdown("---")
                st.markdown(context_text[:1500] + "...")

            for img_path in image_paths_to_show:
                if os.path.exists(img_path):
                    st.image(img_path, caption="关联图片", use_container_width=True)
                else:
                    st.warning(f"⚠️ 图片文件未找到：{img_path}")

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            save_current_session()