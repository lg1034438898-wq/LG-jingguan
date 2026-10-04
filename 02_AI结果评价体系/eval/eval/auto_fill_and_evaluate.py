# -*- coding: utf-8 -*-
"""
景观 RAG 检索质量评估
方案：向量 Top1 作为 silver standard，测混合检索（向量+BM25+RRF）能否命中
"""
import json
import os
import jieba
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

# ==================== 写死路径 ====================
BASE_DIR = r"C:\Users\LG\Desktop\作品集\04_AI结果评价体系建立"
TEST_SET_FILE = os.path.join(BASE_DIR, "eval", "test_set.json")
REPORT_FILE = os.path.join(BASE_DIR, "eval", "retrieval_report.json")

CHROMA_PATH = r"C:\Users\LG\Desktop\作品集\03_景观RAG智能知识库\chroma_db"
COLLECTION_NAME = "landscape_knowledge"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K = 5

# ==================== 加载向量库 ====================
print("=" * 65)
print("🔌 加载资源")
print("=" * 65)
client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
print(f"✅ 向量库：{collection.count()} 条记录")

# 加载所有 chunk 用于 BM25
all_data = collection.get()
all_ids = all_data["ids"]
all_docs = all_data["documents"]
print(f"✅ 加载 {len(all_ids)} 条文档用于 BM25")

tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)
print(f"✅ BM25 索引构建完成")

# ==================== 加载测试集 ====================
with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
    tests = json.load(f)
print(f"✅ 加载测试集：{len(tests)} 道题\n")

# ==================== 评估 ====================
print("=" * 65)
print("📊 检索质量评估")
print("=" * 65)

vector_hit = 0       # 向量检索 Recall@5
hybrid_hit = 0       # 混合检索 Recall@5
vector_mrr = 0.0
hybrid_mrr = 0.0
by_category = {}
details = []

for t in tests:
    query = t["query"]
    category = t["category"]

    # ---- 策略 1：纯向量检索 ----
    vec_res = collection.query(query_texts=[query], n_results=TOP_K)
    vec_ids = vec_res["ids"][0]
    vec_top1 = vec_ids[0] if vec_ids else None

    # ---- 策略 2：向量 + BM25 + RRF 混合 ----
    # 向量 Top 20
    vec_res20 = collection.query(query_texts=[query], n_results=20)
    vec_top20 = vec_res20["ids"][0]

    # BM25 Top 20
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = bm25_scores.argsort()[-20:][::-1]
    bm25_top_ids = [all_ids[i] for i in bm25_top_indices]

    # RRF 融合
    rrf_scores = {}
    k = 60
    for rank, doc_id in enumerate(vec_top20):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_top5 = [doc_id for doc_id, _ in sorted_ids[:TOP_K]]

    # ---- 评估：以向量 Top1 作为 silver standard，看混合检索能不能命中 ----
    if vec_top1:
        # 向量命中（自己和自己比，一定命中）
        v_hit = vec_top1 in vec_ids
        v_rank = vec_ids.index(vec_top1) + 1 if v_hit else None
        if v_hit:
            vector_hit += 1
            vector_mrr += 1 / v_rank

        # 混合检索命中
        h_hit = vec_top1 in hybrid_top5
        h_rank = hybrid_top5.index(vec_top1) + 1 if h_hit else None
        if h_hit:
            hybrid_hit += 1
            hybrid_mrr += 1 / h_rank

        # 分类统计
        by_category.setdefault(category, {
            "total": 0, "vector_hit": 0, "hybrid_hit": 0
        })
        by_category[category]["total"] += 1
        if v_hit:
            by_category[category]["vector_hit"] += 1
        if h_hit:
            by_category[category]["hybrid_hit"] += 1

        details.append({
            "id": t["id"],
            "query": query,
            "category": category,
            "vector_top1": vec_top1,
            "vector_hit": v_hit,
            "hybrid_hit": h_hit,
            "hybrid_rank": h_rank
        })

# ==================== 输出报告 ====================
n = len([d for d in details if d["vector_top1"]])
vec_recall = vector_hit / n if n else 0
hyb_recall = hybrid_hit / n if n else 0
vec_mrr_avg = vector_mrr / n if n else 0
hyb_mrr_avg = hybrid_mrr / n if n else 0

print(f"\n测试集规模 : {n} 条")
print(f"\n{'策略':<20} {'Recall@5':<15} {'MRR':<10}")
print("-" * 65)
print(f"{'纯向量检索':<20} {vec_recall:<15.2%} {vec_mrr_avg:<10.4f}")
print(f"{'混合检索(向量+BM25)':<20} {hyb_recall:<15.2%} {hyb_mrr_avg:<10.4f}")
print("-" * 65)

print("\n📂 分类统计")
print("-" * 65)
print(f"{'类别':<12} {'数量':<8} {'混合检索Recall':<18} {'混合检索MRR'}")
print("-" * 65)
for cat, s in sorted(by_category.items()):
    rate = s["hybrid_hit"] / s["total"]
    print(f"{cat:<12} {s['total']:<8} {rate:<18.1%}")

# 未命中案例
print("\n❌ 混合检索未命中案例（前 5 条）")
print("-" * 65)
fails = [d for d in details if not d["hybrid_hit"]]
for d in fails[:5]:
    print(f"  [{d['id']}] {d['query']}")

# 保存报告
report = {
    "total": n,
    "vector_recall_at_5": vec_recall,
    "hybrid_recall_at_5": hyb_recall,
    "vector_mrr": vec_mrr_avg,
    "hybrid_mrr": hyb_mrr_avg,
    "by_category": {k: {
        "total": v["total"],
        "hybrid_recall": v["hybrid_hit"] / v["total"]
    } for k, v in by_category.items()},
    "details": details
}
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n✅ 报告已保存 → {REPORT_FILE}")