# ChatBot
![image](https://github.com/user-attachments/assets/2974cab1-ed58-4a7a-b620-66a811365773)

## 🧠 中文命名實體識別（NER）
- 模型名稱：**bert-base-chinese-ner**
- 模型下載網址：  
  👉 https://huggingface.co/ckiplab/bert-base-chinese-ner

## 🌍 多語言語句向量模型
- 模型名稱：**paraphrase-multilingual-MiniLM-L12-v2**
- 模型下載網址：  
  👉 https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main

---

## 📁 使用說明

- 將要放入的資料夾加入至 `KM_pool`（儲存至 `chroma_db` 資料夾中）

- 安裝 ChromaDB：
  ```bash
  pip install chromadb

- 加入.last_embedded_files.txt，使用json格式紀錄(能記錄先前已切過的檔案)

- 加入apikey.txt(可至google ai studio申請apikey)
