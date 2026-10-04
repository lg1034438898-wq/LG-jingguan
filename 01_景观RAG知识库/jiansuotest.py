# -*- coding: utf-8 -*-
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import mysql.connector

# ========== 配置 ==========
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1034438898",  # ← 改成你的密码
    "database": "jingguan_rag"
}
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "landscape_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"

# ========== 初始化 ==========
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

conn = mysql.connector.connect(**MYSQL_CONFIG)
cursor = conn.cursor()


# ========== 检索逻辑 ==========
def query_rag(user_query, top_k=5):
    print(f"\n{'=' * 50}\n❓ 问题：{user_query}\n{'=' * 50}")

    # 1. 向量库检索（🌟 修改点1：强制只搜 active）
    results = collection.query(
        query_texts=[user_query],
        n_results=top_k,
        where={"status": "active"}  # 只检索现行有效的规范
    )
    chunk_ids = results["ids"][0]

    # 2. MySQL回查详情（🌟 修改点2：SQL 加上 status='active' 兜底，同时把 doc_name 查出来）
    placeholders = ",".join(["%s"] * len(chunk_ids))
    sql = f"SELECT chunk_id, content, page_number, chunk_type, doc_name FROM chunks WHERE status='active' AND chunk_id IN ({placeholders})"
    cursor.execute(sql, chunk_ids)
    rows = cursor.fetchall()
    row_map = {r[0]: r for r in rows}

    # 3. 按相似度顺序组装并打印
    for i, cid in enumerate(chunk_ids):
        if cid in row_map:
            _, content, page, ctype, doc_name = row_map[cid]  # 🌟 接收 doc_name
            tag = "📷 图片描述" if ctype == "image_description" else ("📊 表格" if ctype == "table" else "📄 文本")
            print(f"\n【片段 {i + 1} | 来源：{doc_name} | 第{page}页 | {tag}】")  # 🌟 打印来源文档
            print(content[:250] + "..." if len(content) > 250 else content)


# ========== 测试 ==========
if __name__ == "__main__":
    test_queries = [
        "植物配置的层次结构是什么？",
        "苗木表里有哪些乔木？",
        "养护质量等级是怎么划分的？"
    ]
    for q in test_queries:
        query_rag(q)

    conn.close()