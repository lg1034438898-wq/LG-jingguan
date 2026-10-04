# -*- coding: utf-8 -*-
"""
把 Excel 测试集转成评估脚本可读的 JSON
路径写死版
"""
import pandas as pd
import json
import os

# 🌟 写死的绝对路径（根据你的实际情况）
BASE_DIR = r"C:\Users\LG\Desktop\作品集\04_AI结果评价体系建立"
EXCEL_FILE = os.path.join(BASE_DIR, "100 道《景观 AI 标准测试集》.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "eval", "test_set.json")

print(f"📁 项目根目录: {BASE_DIR}")
print(f"📄 Excel 文件: {EXCEL_FILE}")

# 检查文件是否存在
if not os.path.exists(EXCEL_FILE):
    print(f"❌ 找不到 Excel 文件：{EXCEL_FILE}")
    exit(1)

# 4 个 sheet 名称
SHEETS = {
    "图纸规范": "图纸规范（25 题）",
    "智能配植": "智能配植（25 题）",
    "方案文本": "方案文本（25 题）",
    "效果图咨询": "效果图咨询（25 题）",
}

all_tests = []

for category, sheet_name in SHEETS.items():
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    except Exception as e:
        print(f"⚠️ 读取 sheet '{sheet_name}' 失败: {e}")
        continue

    df.columns = [c.strip() for c in df.columns]

    for _, row in df.iterrows():
        all_tests.append({
            "id": str(row["编号"]).strip(),
            "query": str(row["测试问题"]).strip(),
            "ground_truth": str(row["标准答案"]).strip(),
            "difficulty": str(row["难度"]).strip(),
            "category": category,
            "correct_chunk_ids": []
        })

# 保存
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_tests, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已转换 {len(all_tests)} 道测试题 → {OUTPUT_FILE}")
print(f"   图纸规范: {sum(1 for t in all_tests if t['category'] == '图纸规范')} 题")
print(f"   智能配植: {sum(1 for t in all_tests if t['category'] == '智能配植')} 题")
print(f"   方案文本: {sum(1 for t in all_tests if t['category'] == '方案文本')} 题")
print(f"   效果图咨询: {sum(1 for t in all_tests if t['category'] == '效果图咨询')} 题")