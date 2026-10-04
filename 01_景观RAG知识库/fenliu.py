# -*- coding: utf-8 -*-
import os
import re
import json
import base64
import requests
import jieba  # 🌟 必须导入
from io import BytesIO
import fitz
import pdfplumber
from PIL import Image
import pytesseract
# ====================== 配置区 ======================
# 🌟 景观专业词典
LANDSCAPE_TERMS = {
    "乔木": ["香樟", "银杏", "桂花", "国槐", "栾树", "樱花", "朴树", "乌桕", "黄连木"],
    "灌木": ["红枫", "紫薇", "绣线菊", "南天竹", "八角金盘", "杜鹃", "栀子"],
    "地被": ["麦冬", "鸢尾", "玉簪", "矾根", "狼尾草", "苔草"],
    "水生植物": ["荷花", "睡莲", "再力花", "黄菖蒲", "旱伞草"],
    "铺装": ["花岗岩", "透水砖", "青石板", "水洗石", "塑木", "防腐木", "卵石"],
    "水景": ["驳岸", "跌水", "涌泉", "镜面水池", "旱喷", "溢水口"],
    "构筑物": ["廊架", "景墙", "树池", "坐凳", "种植池", "挡土墙"],
    "图纸": ["总平面图", "剖面图", "详图", "苗木表", "种植设计图", "竖向设计"],
}
# 🌟 OCR 常见错别字纠错表
OCR_CORRECTION_MAP = {
    "驳崖": "驳岸", "铺妆": "铺装", "乔大": "乔木", "灌大": "灌木",
    "枓": "科", "枠": "架", "術": "木", "圓": "圆", "園": "园",
    "規": "规", "標": "标", "準": "准", "種": "种", "類": "类",
}

# 🌟 植物别名归一化映射 key=标准名 value=别名列表
PLANT_SYNONYM_MAP = {
    "香樟": ["樟树", "芳樟"],
    "乌桕": ["腊子树", "桕子树"],
    "小叶榕": ["细叶榕"],
    "鸡爪槭": ["鸡爪枫"],
    "桂花": ["木犀"],
    "栾树": ["灯笼树"],
    "朴树": ["黄果朴"],
    "麦冬": ["沿阶草"],
    "鸢尾": ["蓝蝴蝶"],
}
# 构建反向别名字典：别名 -> 标准名称
plant_alias_to_std = {}
for std_name, alias_list in PLANT_SYNONYM_MAP.items():
    for alias in alias_list:
        plant_alias_to_std[alias] = std_name

INPUT_DIR = "01_pdf"
TEXT_DIR = "02_text"
IMAGE_DIR = "03_image"
TABLE_DIR = "04_table"
# Tesseract 路径（按你本机实际路径修改）
pytesseract.pytesseract.tesseract_cmd = r'D:\AAA\tesseract.exe'
# Ollama 配置
OLLAMA_URL = "http://127.0.0.1:11434"
VLM_MODEL = "qwen2.5vl:7b"
ENABLE_VLM_TABLE = True
ENABLE_IMAGE_DESC = True
OCR_LANG = "chi_sim+eng"
ZOOM = 2.5
TEXT_THRESHOLD = 50

for d in [INPUT_DIR, TEXT_DIR, IMAGE_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

# ====================== 初始化 jieba 专业词典 ======================
total_terms = sum(len(v) for v in LANDSCAPE_TERMS.values())
for category, terms in LANDSCAPE_TERMS.items():
    for term in terms:
        jieba.add_word(term)
# 将标准植物名也加入jieba词库
for std in PLANT_SYNONYM_MAP.keys():
    jieba.add_word(std)
for alias in plant_alias_to_std.keys():
    jieba.add_word(alias)

print(f"✅ 已加载景观专业词汇 {total_terms} 条")

# ====================== 植物名称归一化函数（基于jieba分词，避免子串误替换） ======================
def plant_name_normalize(text: str) -> str:
    """植物别名归一化，使用jieba分词只替换完整词汇，防止部分字符误匹配"""
    if not text:
        return text
    word_list = jieba.lcut(text)
    output_words = []
    for w in word_list:
        if w in plant_alias_to_std:
            output_words.append(plant_alias_to_std[w])
        else:
            output_words.append(w)
    return "".join(output_words)

# ====================== 统一 VLM 调用 ======================
def vlm_generate(prompt, image_b64=None, timeout=300):
    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    if image_b64:
        payload["images"] = [image_b64]
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "").strip()
    except requests.exceptions.HTTPError as e:
        return f"[VLM HTTP错误] {e} | 响应: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        return "[VLM失败] 无法连接Ollama，请确认 ollama serve 已启动"
    except requests.exceptions.Timeout:
        return "[VLM失败] 请求超时"
    except Exception as e:
        return f"[VLM失败] {type(e).__name__}: {e}"

# ====================== OCR（含专业纠错 + 植物别名归一化） ======================
def ocr_extract_page(fitz_page, zoom=2.5):
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = fitz_page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert('L')
        img = img.point(lambda p: 0 if p < 190 else 255)
        text = pytesseract.image_to_string(img, lang=OCR_LANG, config=r'--oem 3 --psm 6')
        # 🌟 应用 OCR 纠错表
        for wrong, right in OCR_CORRECTION_MAP.items():
            text = text.replace(wrong, right)
        # 🌟 OCR结果执行植物别名归一化
        text = plant_name_normalize(text)
        return text
    except Exception as e:
        print(f"⚠️ OCR失败: {e}")
        return ""

# ====================== VLM 表格提取 ======================
def vlm_extract_table_from_page(fitz_page, page_num):
    try:
        mat = fitz.Matrix(1.2, 1.2)
        pix = fitz_page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompt = """任务：提取图片中的表格。规则：
1、只输出markdown表格，禁止任何解释；
2、完整保留表头、所有行、全部单元格；
3、无表格只输出：__NO_TABLE__"""
        res = vlm_generate(prompt, image_b64=img_b64)
        if res.strip() == "__NO_TABLE__" or res.startswith("[VLM"):
            return ""
        if "|" not in res:
            return ""
        return f"\n## 第{page_num}页【VLM表格】\n{res}"
    except Exception as e:
        print(f"⚠️ 第{page_num}页VLM表格失败: {e}")
        return ""

# ====================== 图片描述（专业提示词） ======================
def describe_image(image_path, context_text=""):
    try:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1280:
            img.thumbnail((1280, 1280))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompt = """你是景观设计与施工图领域的资深专家。请用**景观行业标准术语**描述这张图纸。
【硬性规则】
1. 禁止使用口语化词汇。必须用专业术语：
   - 不说"大树/小树"，要说"乔木/灌木/地被/水生植物"
   - 不说"水边"，要说"驳岸/水池/跌水/涌泉"
   - 不说"地板"，要说"铺装/花岗岩/透水砖/青石板"
2. 图纸类型判断（必须明确写出）：总平面图 / 剖面图 / 详图 / 苗木表 / 种植设计图 / 竖向设计图 / 效果图。
3. 提取信息维度：
   - 植物类：植物品种（如"香樟""银杏""红枫"）、冠幅、株高、种植密度
   - 硬景类：材料名称（如"花岗岩""防腐木"）、尺寸标注、构造层次
   - 水景类：驳岸形式、水深、铺装材料
   - 文字类：图名、编号、尺寸数字、规范代号（如CJJ/T 287）
4. 结合上下文辅助理解：{context_text[:150]}
【输出格式】
直接输出描述，不要任何客套话（如"这张图展示了"）。字数控制在200字以内。"""
        return vlm_generate(prompt, image_b64=img_b64)
    except Exception as e:
        return f"[图片描述生成失败: {e}]"

# ====================== 文本清洗（保护专业词汇） ======================
def clean_text(text):
    # 🌟 保护规范编号和植物拉丁名
    protected = {}
    def protect(match):
        key = f"__PROTECT_{len(protected)}__"
        protected[key] = match.group(0)
        return key
    text = re.sub(r'[A-Z]{2,}(?:/T)?\s*\d+[-—]\d{4}', protect, text)  # 规范编号
    text = re.sub(r'[A-Z][a-z]+\s+[a-z]+(?:\s+[a-z]+)?', protect, text)  # 拉丁名
    # 原有的清洗逻辑
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'www\.biaozhun\.org', '', text)
    text = re.sub(r'www\.gb688\.com', '', text)
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'共\s*\d+\s*页', '', text)
    text = re.sub(r'([^。！？；：\n])\n(?!\d+\.\d+)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 还原保护的专业词汇
    for key, val in protected.items():
        text = text.replace(key, val)
    return text.strip()

# ====================== 提取文本+表格 ======================
def extract_text_and_collect_tables(pdf_path, out_name):
    full_text, page_meta, table_md_parts = [], [], []
    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            raw_txt = page.extract_text() or ""
            is_ocr_page = False
            if len(raw_txt.strip()) < TEXT_THRESHOLD:
                print(f"第{page_num}页启用OCR")
                is_ocr_page = True
                fitz_p = doc.load_page(page_idx)
                raw_txt = ocr_extract_page(fitz_p, zoom=ZOOM)
                if ENABLE_VLM_TABLE:
                    vlm_table = vlm_extract_table_from_page(fitz_p, page_num)
                    if vlm_table:
                        table_md_parts.append(vlm_table)
            else:
                tables = page.extract_tables()
                if tables:
                    for tid, tbl in enumerate(tables):
                        if len(tbl) < 2:
                            continue
                        table_md_parts.append(f"\n## 第{page_num}页【矢量表格】{tid + 1}\n")
                        header = [str(c).strip().replace("\n", " ") if c else "" for c in tbl[0]]
                        table_md_parts.append("| " + " | ".join(header) + " |")
                        table_md_parts.append("|" + "|".join(["---"] * len(header)) + "|")
                        for row in tbl[1:]:
                            cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
                            table_md_parts.append("| " + " | ".join(cells) + " |")
            # 矢量页面执行植物别名归一化
            normalized_txt = plant_name_normalize(raw_txt)
            cleaned = clean_text(normalized_txt)
            if cleaned:
                full_text.append(f"\n===== 第 {page_num} 页 =====\n{cleaned}")
                page_meta.append({"page": page_num, "char_count": len(cleaned), "is_ocr_page": is_ocr_page})
    txt_path = os.path.join(TEXT_DIR, f"{out_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    meta_path = os.path.join(TEXT_DIR, f"{out_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(page_meta, f, ensure_ascii=False, indent=2)
    print(f"[文本] → {txt_path}（{len(page_meta)} 页）")
    return txt_path, table_md_parts

# ====================== 提取图片 ======================
def extract_images(pdf_path, out_name):
    sub_dir = os.path.join(IMAGE_DIR, out_name)
    os.makedirs(sub_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    records = []
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        page_area = page.rect.width * page.rect.height
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            if rect.width < 50 or rect.height < 50:
                continue
            if (rect.width * rect.height) / page_area > 0.9:
                continue
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = os.path.join(sub_dir, f"page{page_num}_img{img_idx}.png")
            pix.save(img_path)
            pix = None
            clip = fitz.Rect(rect.x0 - 200, rect.y0 - 50, rect.x1 + 200, rect.y1 + 50)
            context = page.get_text(clip=clip).strip()[:400]
            records.append({"page": page_num, "image_path": img_path, "context_text": context})
    doc.close()
    print(f"[图片] 已提取 {len(records)} 张 → {sub_dir}")
    return records, sub_dir

# ====================== 保存表格md（表格同样执行植物别名归一化） ======================
def save_tables_md(out_name, table_md_parts):
    # 表格每段执行植物名称归一化
    normalized_table_list = [plant_name_normalize(item) for item in table_md_parts]
    md_path = os.path.join(TABLE_DIR, f"{out_name}_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {out_name} 表格提取结果\n" + "\n".join(normalized_table_list))
    print(f"[表格] {len(normalized_table_list)} 段 → {md_path}")
    return md_path

# ====================== 主流程 ======================
def process_pdf(pdf_path):
    out_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # 检查是否已经解析过。如果02文件夹里已经有这个文档的txt，就跳过
    txt_path = os.path.join(TEXT_DIR, f"{out_name}.txt")
    if os.path.exists(txt_path):
        print(f"⏭️ 跳过（已解析过）: {out_name}")
        return
    print(f"\n===== 处理：{out_name} =====")
    _, table_md_parts = extract_text_and_collect_tables(pdf_path, out_name)
    save_tables_md(out_name, table_md_parts)
    image_records, img_sub_dir = extract_images(pdf_path, out_name)
    if ENABLE_IMAGE_DESC and image_records:
        desc_md_path = os.path.join(img_sub_dir, "image_desc.md")
        md_lines = [f"# {out_name} 图片VLM描述\n"]
        for rec in image_records:
            print(f"  正在生成描述：{rec['image_path']}")
            desc = describe_image(rec["image_path"], rec["context_text"])
            md_lines.append(f"## 第{rec['page']}页 | {os.path.basename(rec['image_path'])}")
            md_lines.append(f"> 页面上下文：{rec['context_text']}")
            md_lines.append(f"> VLM描述：{desc}\n")
        with open(desc_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[图片描述] → {desc_md_path}")

def main():
    pdfs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"⚠️ {INPUT_DIR}/ 下没有PDF")
        return
    for pdf in pdfs:
        process_pdf(os.path.join(INPUT_DIR, pdf))
    print("\n✅ 全部完成")

if __name__ == "__main__":
    main()