# -*- coding: utf-8 -*-
"""
将 05/all_chunks.json 中的数据同时写入 MySQL 和 Chroma 向量库
"""
import os
import json
import chromadb
import mysql.connector
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==================== 配置区（必须修改） ====================
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1034438898",      # ← 【必须修改】改成你本机真实的MySQL密码
    "database": "jingguan_RAG"      # ← 确保该数据库已创建
}

CHROMA_PATH = "./chroma_db"          # 向量库持久化目录
CHUNK_FILE = os.path.join("05_rag_output", "all_chunks.json")  # 输入文件路径
COLLECTION_NAME = "landscape_knowledge"              # Chroma集合名称
EMBEDDING_MODEL = "BAAI/bge-m3"                      # 嵌入模型
BATCH_SIZE = 100                     # Chroma分批写入大小

# ==================== 初始化检查 ====================
if not os.path.exists(CHUNK_FILE):
    print(f"❌ 找不到文件：{CHUNK_FILE}")
    print("请先运行 merge_chunks.py 生成 05/all_chunks.json")
    exit(1)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if not chunks:
    print("❌ 05/all_chunks.json 为空，请检查数据")
    exit(1)

print(f"✅ 成功读取 {len(chunks)} 个 chunk")


# ==================== 1. 写入 MySQL ====================
def write_to_mysql():
    print("\n[1/2] 正在写入 MySQL...")
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
    except mysql.connector.Error as e:
        print(f"❌ MySQL 连接失败: {e}")
        print("请检查密码是否正确，MySQL 服务是否启动，数据库是否已创建。")
        exit(1)

    cursor = conn.cursor()

    # 建表：documents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INT PRIMARY KEY AUTO_INCREMENT,
        doc_name VARCHAR(255) NOT NULL UNIQUE,
        source_file VARCHAR(255),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)

    # 建表：chunks
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id VARCHAR(128) PRIMARY KEY,
        doc_id INT,
        doc_name VARCHAR(255),
        page_number INT,
        chunk_type VARCHAR(32),
        content TEXT,
        image_path VARCHAR(255),
        status VARCHAR(20) DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    conn.commit()

    # 插入 documents 表并获取 doc_id 映射
    doc_names = sorted(set(c["metadata"]["doc_name"] for c in chunks))
    doc_id_map = {}
    for name in doc_names:
        cursor.execute(
            "INSERT IGNORE INTO documents (doc_name, source_file) VALUES (%s, %s)",
            (name, name + ".pdf")
        )
        conn.commit()
        cursor.execute("SELECT doc_id FROM documents WHERE doc_name=%s", (name,))
        doc_id_map[name] = cursor.fetchone()[0]

    # 插入 chunks 表
    insert_sql = """
        INSERT IGNORE INTO chunks
        (chunk_id, doc_id, doc_name, page_number, chunk_type, content, image_path, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    mysql_data = []
    for c in chunks:
        meta = c["metadata"]
        mysql_data.append((
            c["chunk_id"],
            doc_id_map[meta["doc_name"]],
            meta["doc_name"],
            meta.get("page", 0),
            meta.get("type", "text"),
            c["content"],
            meta.get("image_path", ""),
            meta.get("status", "active")  # 🌟 加上这个
        ))

    # 批量插入，每100条提交一次
    for i in range(0, len(mysql_data), BATCH_SIZE):
        cursor.executemany(insert_sql, mysql_data[i:i + BATCH_SIZE])
        conn.commit()

    cursor.close()
    conn.close()
    print(f"✅ MySQL 写入完成，共 {len(mysql_data)} 条")


# ==================== 2. 写入 Chroma ====================
def write_to_chroma():
    print("\n[2/2] 正在写入 Chroma 向量库...")
    print("提示：首次运行会自动下载嵌入模型（约2GB），请耐心等待...")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"❌ Chroma 初始化失败: {e}")
        exit(1)

    # 🌟 新增：获取当前 Chroma 已有的所有 chunk_id
    print("  正在获取现有向量库记录...")
    existing_ids = set()
    try:
        existing_data = collection.get()
        existing_ids = set(existing_data['ids'])
        print(f"  当前向量库已有 {len(existing_ids)} 条记录")
    except Exception as e:
        print(f"  获取现有向量失败（可能是首次运行）: {e}")

    # 🌟 新增：过滤掉已经存在的 chunk
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("✅ 没有发现新数据，所有 chunk 均已入库！")
        return

    print(f"  发现 {len(new_chunks)} 条新数据，准备向量化...")

    # 分批写入（使用 new_chunks 而不是原来的 chunks）
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        metadatas = []

        for c in batch:
            meta = c["metadata"]
            m = {
                "doc_name": str(meta.get("doc_name", "")),
                "page": int(meta.get("page", 0)),
                "type": str(meta.get("type", "text")),
                "status": str(meta.get("status", "active"))  # 🌟 新增这行，默认active
            }
            if meta.get("image_path"):
                m["image_path"] = str(meta["image_path"])
            metadatas.append(m)

        # 注意：这里依然使用 upsert，防止刚好有重复ID
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Chroma 入库进度: {min(i + BATCH_SIZE, len(new_chunks))}/{len(new_chunks)}")

    print(f"✅ Chroma 更新完成，新增 {len(new_chunks)} 条，当前总数 {collection.count()} 条向量")


# ==================== 主流程 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行数据双写 (MySQL + Chroma)")
    print("=" * 50)
    write_to_mysql()
    write_to_chroma()
    print("\n✅ 全部完成！")