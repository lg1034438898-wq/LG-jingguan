# -*- coding: utf-8 -*-
import os
import re
import json

TEXT_DIR = "02_text"
IMAGE_DIR = "03_image"
TABLE_DIR = "04_table"
OUTPUT_DIR = "05_rag_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_chunks.json")

all_chunks = []
chunk_counter = 0

# ============ 1. 处理02文本 ============
def process_text():
    global chunk_counter
    for filename in os.listdir(TEXT_DIR):
        if not filename.endswith(".txt"):
            continue
        doc_name = filename.replace(".txt", "")
        filepath = os.path.join(TEXT_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 按页码分块
        pages = re.split(r"===== 第 (\d+) 页 =====", content)
        # pages结构: ['', '1', '页1内容', '2', '页2内容', ...]
        for i in range(1, len(pages), 2):
            page_num = int(pages[i])
            page_text = pages[i+1].strip()
            if not page_text:
                continue

            # 按段落切分，每块400字左右
            paragraphs = page_text.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                # 在循环处理段落时，动态生成带有元数据的 chunk 内容
                # 假设你解析出的章节标题存在变量 section_title 里（如果没有提取章节，可以用文档名代替）
                meta_header = f"【文档：{doc_name} | 页码：{page_num}】\n"

                if len(current_chunk) + len(para) > 300 and current_chunk:
                    chunk_counter += 1
                    all_chunks.append({
                        "chunk_id": f"text_{doc_name}_p{page_num}_c{chunk_counter}",
                        "content": meta_header + current_chunk.strip(),  # 🌟 注入元数据
                        "metadata": {
                            "doc_name": doc_name,
                            "page": page_num,
                            "type": "text",
                            "source": filename
                        }
                    })
                    # 注意：这里也要同步修改 current_chunk 的拼接逻辑（加上重叠和元数据）
                    current_chunk = current_chunk[-50:] + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            if current_chunk.strip():
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"text_{doc_name}_p{page_num}_c{chunk_counter}",
                    "content": current_chunk.strip(),
                    "metadata": {
                        "doc_name": doc_name,
                        "page": page_num,
                        "type": "text",
                        "source": filename
                    }
                })

# ============ 2. 处理03图片描述 ============
def process_images():
    global chunk_counter
    for root, dirs, files in os.walk(IMAGE_DIR):
        if "image_desc.md" not in files:
            continue
        doc_name = os.path.basename(root)
        filepath = os.path.join(root, "image_desc.md")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 "## 第X页 | 图片名" 切分
        pattern = r"## 第(\d+)页 \| (page\d+_img\d+\.png)"
        matches = list(re.finditer(pattern, content))
        for idx, match in enumerate(matches):
            page_num = int(match.group(1))
            img_name = match.group(2)
            start = match.end()
            end = matches[idx+1].start() if idx + 1 < len(matches) else len(content)
            block = content[start:end].strip()

            # 提取VLM描述
            desc_match = re.search(r"> VLM描述：(.*?)(?=\n##|\Z)", block, re.DOTALL)
            if desc_match:
                desc = desc_match.group(1).strip()
                # 过滤掉失败描述
                if "[图片描述生成失败" in desc or "VLM失败" in desc:
                    desc = "该图片未能生成有效描述。"
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"img_{doc_name}_p{page_num}_{img_name}",
                    "content": desc,
                    "metadata": {
                        "doc_name": doc_name,
                        "page": page_num,
                        "type": "image_description",
                        "image_path": os.path.join(root, img_name),
                        "source": "image_desc.md"
                    }
                })

# ============ 3. 处理04表格 ============
def process_tables():
    global chunk_counter
    for filename in os.listdir(TABLE_DIR):
        if not filename.endswith(".md"):
            continue
        doc_name = filename.replace("_tables.md", "")
        filepath = os.path.join(TABLE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"## 第(\d+)页"
        matches = list(re.finditer(pattern, content))
        for idx, match in enumerate(matches):
            page_num = int(match.group(1))
            start = match.end()
            end = matches[idx+1].start() if idx + 1 < len(matches) else len(content)
            table_md = content[start:end].strip()

            if table_md:
                # 🌟 新增优化逻辑：拆分长表格
                lines = table_md.split("\n")
                header = lines[0] + "\n" + lines[1] if len(lines) > 1 else ""
                body_rows = lines[2:]
                chunk_size_rows = 15  # 每15行拆成一个chunk

                # 如果表格行数较少，直接作为一个chunk
                if len(body_rows) <= chunk_size_rows:
                    chunk_counter += 1
                    all_chunks.append({
                        "chunk_id": f"table_{doc_name}_p{page_num}_c{chunk_counter}",
                        "content": table_md,
                        "metadata": {
                            "doc_name": doc_name,
                            "page": page_num,
                            "type": "table",
                            "source": filename
                        }
                    })
                else:
                    # 如果表格太长，按15行拆分，并在每个chunk前加上表头
                    for j in range(0, len(body_rows), chunk_size_rows):
                        chunk_counter += 1
                        chunk_content = header + "\n" + "\n".join(body_rows[j:j+chunk_size_rows])
                        all_chunks.append({
                            "chunk_id": f"table_{doc_name}_p{page_num}_c{chunk_counter}_part{j//chunk_size_rows + 1}",
                            "content": chunk_content,
                            "metadata": {
                                "doc_name": doc_name,
                                "page": page_num,
                                "type": "table",
                                "source": filename
                            }
                        })

# ============ 主流程 ============
if __name__ == "__main__":
    process_text()
    process_images()
    process_tables()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 整合完成，共生成 {len(all_chunks)} 个chunk")
    print(f"  文本chunk: {sum(1 for c in all_chunks if c['metadata']['type']=='text')}")
    print(f"  图片描述chunk: {sum(1 for c in all_chunks if c['metadata']['type']=='image_description')}")
    print(f"  表格chunk: {sum(1 for c in all_chunks if c['metadata']['type']=='table')}")