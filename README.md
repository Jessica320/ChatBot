# RAG Chatbot based on LangChain
![image](https://github.com/user-attachments/assets/2974cab1-ed58-4a7a-b620-66a811365773)

## 🧠 中文命名實體識別（NER）
- 模型名稱：**bert-base-chinese-ner**
- 模型下載網址： https://huggingface.co/ckiplab/bert-base-chinese-ner

## 🌍 多語言語句向量模型
- 模型名稱：**paraphrase-multilingual-MiniLM-L12-v2**
- 模型下載網址： https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main

## 📁 使用說明

- 將要放入的資料夾加入至 `KM_pool`（儲存至 `chroma_db` 資料夾中）

- 安裝 ChromaDB：
  ```bash
  pip install chromadb

- 加入 .last_embedded_files.txt
  使用 JSON 格式 紀錄（可記錄先前已切割過的檔案，避免重複處理）

- 加入 apikey.txt
  可至 Google AI Studio 申請並填入 API Key

---
# Translate in English

## 🧠 Chinese Named Entity Recognition (NER)
- Model: **bert-base-chinese-ner**
- Download link: https://huggingface.co/ckiplab/bert-base-chinese-ner

## 🌍 Multilingual Sentence Embedding Model
- Model: **paraphrase-multilingual-MiniLM-L12-v2**
- Download link: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main

## 📁 Usage Instructions

- Add the folder containing the documents to `KM_pool` (files will be stored in the `chroma_db` directory)

- Install ChromaDB:
  ```bash
  pip install chromadb
  
- Add .last_embedded_files.txt
  Use JSON format to record previously processed (chunked) files to avoid duplication

- Add apikey.txt
  You can obtain an API key from Google AI Studio and place it in this file
