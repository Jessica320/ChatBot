import os
import shutil
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from docx import Document as DocxReader  # 導入讀取 docx 的庫

# 設置參數
MODEL_PATH = "./paraphrase-multilingual-MiniLM-L12-v2"  # 本地嵌入模型路徑
PDF_DIR = "./KM_pool"  # PDF 和 DOCX 檔案目錄
CHROMA_PATH = "./chroma_db"  # Chroma 儲存路徑
CHUNK_SIZE = 500  # 每個 chunk 的目標字符數
CHUNK_OVERLAP = 100  # chunk 間的重疊字符數

# 初始化嵌入函數
def init_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_PATH)

# 提取 PDF 文字並清理
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            # 清理多餘換行和空白
            text = " ".join(text.split())
            if text:
                text_by_page.append({"page_num": page_num, "text": text})
        return text_by_page
    except Exception as e:
        print(f"讀取PDF時發生錯誤: {e}")
        return []

# 提取 DOCX 文字並清理
def extract_text_from_docx(docx_path):
    try:
        document = DocxReader(docx_path)
        full_text = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                full_text.append({"text": text})
        return full_text
    except Exception as e:
        print(f"讀取DOCX時發生錯誤: {e}")
        return []

# 分割文字（帶重疊）
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        # 確保不切斷單詞
        if end < text_length:
            while end > start and text[end] not in " .!?":
                end -= 1
            if end == start:
                end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_length else text_length

    return chunks

# 處理並嵌入單個檔案 (PDF 或 DOCX)
def process_file(file_path, collection):
    file_name = os.path.basename(file_path)
    print(f"處理檔案：{file_name}")
    all_documents = []
    all_metadatas = []
    all_ids = []
    total_chunks = 0

    if file_path.lower().endswith(".pdf"):
        pages = extract_text_from_pdf(file_path)
        if not pages:
            print(f"警告：{file_name} 無內容或讀取失敗，跳過")
            return
        for page in pages:
            page_num = page["page_num"]
            text = page["text"]
            chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_documents.append(f"**檔案名稱：{file_name}**\n\n內容：{chunk}") # 只儲存原始文本 chunk
                all_metadatas.append({
                    "source": file_name,
                    "page_num": page_num,
                    "chunk_id": i,
                    "file_name": file_name
                })
                all_ids.append(f"{file_name}_page{page_num}_{i}")
            total_chunks += len(chunks)
            print(f"已分割 {file_name} 頁 {page_num}，共 {len(chunks)} 個片段")

    elif file_path.lower().endswith(".docx"):
        paragraphs = extract_text_from_docx(file_path)
        if not paragraphs:
            print(f"警告：{file_name} 無內容或讀取失敗，跳過")
            return
        for i, paragraph_data in enumerate(paragraphs):
            text = paragraph_data["text"]
            chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for j, chunk in enumerate(chunks):
                all_documents.append(f"檔案名稱：{file_name}\n內容：{chunk}") # 只儲存原始文本 chunk
                all_metadatas.append({
                    "source": file_name,
                    "paragraph": i + 1,
                    "chunk_id": j,
                    "file_name": file_name
                })
                all_ids.append(f"{file_name}_para{i}_{j}")
            total_chunks += len(chunks)
            print(f"已分割 {file_name} 段落 {i+1}，共 {len(chunks)} 個片段")

    # 批量嵌入
    if all_documents:
        collection.add(
            documents=all_documents,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"已嵌入 {file_name}，共 {total_chunks} 個文本片段")
    else:
        print(f"{file_name} 無有效內容可嵌入")

# 處理 PDF 和 DOCX 檔案
def process_pdfs_and_docx(pdf_dir, collection):
    all_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith((".pdf", ".docx"))]
    if not all_files:
        print(f"警告：目錄 {pdf_dir} 中找不到 PDF 或 DOCX 檔案")
        return

    for file_name in all_files:
        file_path = os.path.join(pdf_dir, file_name)
        process_file(file_path, collection)

def main():
    # 檢查路徑
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"嵌入模型路徑 {MODEL_PATH} 不存在")
    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f"PDF/DOCX 目錄 {PDF_DIR} 不存在")

    # 清空並重建 Chroma 路徑
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"已刪除舊的 ChromaDB 資料夾：{CHROMA_PATH}")
    os.makedirs(CHROMA_PATH)
    print(f"已創建新的 ChromaDB 資料夾：{CHROMA_PATH}")

    # 初始化 Chroma 客戶端
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = "pdf_docx_collection"

    # 創建新集合
    embedding_function = init_embedding_function()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    print(f"已創建新的集合：{collection_name}")

    # 處理 PDF 和 DOCX 並建立向量資料庫
    process_pdfs_and_docx(PDF_DIR, collection)

    # 檢查集合狀態
    print(f"集合 {collection_name} 現有 {collection.count()} 個嵌入向量")
    print(f"向量資料庫已重建並儲存至 {CHROMA_PATH}")

if __name__ == "__main__":
    main()