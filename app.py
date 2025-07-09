import streamlit as st
from chatbot import create_rag_chain, ask_question, process_source_documents, load_api_key, API_KEY_FILE
import os
import subprocess
import time
import json
import warnings
import sys  # 導入 sys 模組

# 忽略非致命警告
warnings.filterwarnings("ignore")

# 設定參數 (與 embedding.py 相同)
PDF_DIR = "./KM_pool"
CHROMA_PATH = "./chroma_db"
EMBEDDING_SCRIPT = "embedding.py"  # embedding 腳本的名稱
LAST_EMBEDDED_FILES_FILE = ".last_embedded_files.txt"

# 主要藍色 (參考 KGI Bank 圖片)
primary_blue = "#0047AB"
light_blue = "#ADD8E6"
text_gray = "#333333"
background_light = "#F5F5F5"

# 設定 Streamlit 頁面
st.set_page_config(page_title="𝐝𝐨𝐜.𝐀𝐈", page_icon="📄")

# 設定主題色彩
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {primary_blue};
        --background-color: {background_light};
        --secondary-background-color: white;
        --text-color: {text_gray};
        --font-family: sans-serif;
    }}
    body {{
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }}
    .st-emotion-cache-r421ms {{ /* 標題樣式 */
        color: var(--primary-color);
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }}
    .st-emotion-cache-10pwf3t {{ /* 副標題/介紹文字樣式 */
        color: var(--text-color);
        font-size: 1.1em;
        margin-bottom: 1em;
        line-height: 1.5;
    }}
    .st-emotion-cache-16txtl3 {{ /* 輸入框樣式 */
        border-radius: 5px;
        border-color: var(--primary-color);
        box-shadow: 1px 1px 3px #cccccc;
    }}
    .st-emotion-cache-676k2g {{ /* 聊天訊息使用者樣式 */
        background-color: {light_blue};
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.6em;
        margin-bottom: 0.4em;
    }}
    .st-emotion-cache-1w011k9 {{ /* 聊天訊息助理樣式 */
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 8px;
        padding: 0.6em;
        margin-bottom: 0.4em;
        box-shadow: 1px 1px 2px #e0e0e0;
    }}
    .st-emotion-cache-10fy7yf {{ /* 展開器樣式 */
        border-radius: 5px;
        border-color: var(--primary-color);
        box-shadow: 1px 1px 2px #cccccc;
    }}
    .sidebar-title {{
        color: var(--primary-color);
        font-weight: bold;
        margin-bottom: 0.5em;
        font-size: 1.8em;
    }}
    .sidebar-content {{
        color: var(--text-color);
        line-height: 1.4;
        font-size: 1.2em;
    }}
    .new-chat-button {{
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.4em 0.8em;
        margin-bottom: 0.4em;
        cursor: pointer;
        box-shadow: 1px 1px 2px #cccccc;
    }}
    .new-chat-button:hover {{
        background-color: #0056b3;
    }}
    .chat-history-item {{
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 5px;
        padding: 0.4em;
        margin-bottom: 0.2em;
        cursor: pointer;
    }}
    .chat-history-item:hover {{
        background-color: #f0f0f0;
    }}
    .suggestion-button {{
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.3em 0.6em;
        margin-right: 0.3em;
        margin-bottom: 0.3em;
        cursor: pointer;
        font-size: 0.9em;
        box-shadow: 1px 1px 1px #cccccc;
    }}
    .suggestion-button:hover {{
        background-color: #0056b3;
    }}
    /* 調整「顯示參考文件」按鈕樣式 */
    .reference-button {{
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 3px;
        padding: 0.2em 0.4em;
        font-size: 0.8em;
        cursor: pointer;
        box-shadow: 1px 1px 1px #cccccc;
        margin-top: 0.3em;
    }}
    .reference-button:hover {{
        background-color: #0056b3;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 初始化會話狀態
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = None
if "all_chat_history" not in st.session_state:
    st.session_state["all_chat_history"] = {}
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "summary_button_clicked" not in st.session_state:
    st.session_state["summary_button_clicked"] = False
if "show_suggestion_button" not in st.session_state:
    st.session_state["show_suggestion_button"] = True
if "show_sources" not in st.session_state:
    st.session_state["show_sources"] = False
if "current_sources" not in st.session_state:
    st.session_state["current_sources"] = []
if "show_reference" not in st.session_state:
    st.session_state["show_reference"] = False  # 用來控制是否顯示參考文件

def get_current_file_list(directory):
    """獲取目前目錄下所有 .pdf 和 .docx 檔案的排序列表。"""
    return sorted([f for f in os.listdir(directory) if f.lower().endswith((".pdf", ".docx"))])

def load_last_embedded_files():
    """從檔案載入上次成功 embedding 的檔案列表。"""
    try:
        with open(LAST_EMBEDDED_FILES_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

def save_last_embedded_files(file_list):
    """將目前成功 embedding 的檔案列表儲存到檔案。"""
    with open(LAST_EMBEDDED_FILES_FILE, "w") as f:
        json.dump(file_list, f)

def run_embedding_script(changed_files_message=""):
    """執行 embedding.py 腳本，確保使用目前的 Python 環境。"""
    try:
        # 使用 sys.executable 獲取目前 Python 解釋器的路徑 (在 venv 中運行時會指向 venv 的 python)
        subprocess.run([sys.executable, EMBEDDING_SCRIPT], check=True)
        time.sleep(3)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"錯誤：執行 {EMBEDDING_SCRIPT} 失敗：{e}")
        return False
    except FileNotFoundError:
        st.error(f"錯誤：找不到 {EMBEDDING_SCRIPT} 檔案，請確保它與 app.py 在同一目錄下。")
        return False

# 側邊欄：顯示對話紀錄和檔案列表
with st.sidebar:
    st.markdown("<p class='sidebar-title'>對話記錄</p>", unsafe_allow_html=True)
    if st.button("💬  新增對話", key="new_chat", help="開始新的對話"):
        st.session_state.update(
            messages=[],
            chat_history=[],
            current_chat_id=None,
            summary_button_clicked=False,
            show_suggestion_button=True,
            show_sources=False,
            current_sources=[],
            show_reference=False
        )
        st.rerun()

    for chat_id, chat_info in st.session_state["all_chat_history"].items():
        if st.button(chat_info.get("title", f"對話 {chat_id}"), key=f"chat_{chat_id}"):
            st.session_state.update(
                current_chat_id=chat_id,
                messages=[{"role": m["type"].lower(), "content": m["content"]} for m in chat_info["history"]],
                chat_history=chat_info["history"],
                summary_button_clicked=False,
                show_suggestion_button=True,
                show_sources=False,
                current_sources=[],
                show_reference=False
            )
            st.rerun()

    st.markdown("---")
    st.markdown("<p class='sidebar-title'>📂 文件列表：</p>", unsafe_allow_html=True)
    all_files = get_current_file_list(PDF_DIR)
    for i, file in enumerate(all_files):
        st.markdown(f"<p class='sidebar-content'>{i+1}. {file}</p>", unsafe_allow_html=True)

# 主介面
st.title("𝐝𝐨𝐜.𝐀𝐈")
st.markdown("<p style='font-size: 1.1em; line-height: 1.5;'>檢索檔案內文，解答您的疑問。</p>", unsafe_allow_html=True)

# 載入 API 金鑰
api_key = load_api_key(API_KEY_FILE)

if api_key:
    current_files = get_current_file_list(PDF_DIR)
    last_embedded_files = load_last_embedded_files()

    added_files = [f for f in current_files if f not in last_embedded_files]
    removed_files = [f for f in last_embedded_files if f not in current_files]

    changed = False
    change_messages = []

    if added_files:
        change_messages.append(f"新增文件：{', '.join(added_files)}")
        changed = True
    if removed_files:
        change_messages.append(f"刪除文件：{', '.join(removed_files)}")
        changed = True

    if changed:
        changes_description = "；".join(change_messages)
        st.info(f"📄 偵測到文件變更 ({changes_description})。")

        # 創建一個佔位符來顯示處理訊息
        processing_placeholder = st.empty()
        processing_placeholder.info("正在處理您的檔案中...")

        if run_embedding_script(changes_description):
            processing_placeholder.success("✅ 知識庫更新完成") # 將成功訊息顯示在佔位符中
            save_last_embedded_files(current_files)
            time.sleep(3) # 讓成功訊息停留一會兒
            st.session_state["rag_chain"] = None  # 重新載入 RAG 鏈
        else:
            processing_placeholder.error("⚠️ 知識庫更新失敗，可能無法反映最新的文件變更。")
    else:
        st.info("📚 文件沒有變更，使用現有知識庫。")

    # 初始化 RAG 鏈
    @st.cache_resource(show_spinner="正在載入知識和模型...")
    def load_rag_chain(api_key):
        return create_rag_chain(api_key)

    if st.session_state["rag_chain"] is None:
        st.session_state["rag_chain"] = load_rag_chain(api_key)

    if st.session_state["rag_chain"]:
        st.markdown("---")
        st.markdown("**💡  功能提示**")
        st.markdown("<p style='line-height: 1.5;'>我是您的專業知識助手，可以幫您查找文件資訊、解釋內容或進行摘要，請隨時提出您的問題。</p>", unsafe_allow_html=True)
        st.markdown("---")

        # 顯示建議按鈕 (移到聊天輸入框之前)
        if st.session_state["show_suggestion_button"]:
            st.markdown("<p style='font-size: 0.9em; color: #777;'>不知道該問什麼？點擊按鈕快速了解功能。</p>", unsafe_allow_html=True)
            suggestion_button = st.button("你能提供甚麼服務跟功能?", key="summary_suggestion")
            if suggestion_button and not st.session_state["summary_button_clicked"]:
                st.session_state["summary_button_clicked"] = True
                st.session_state["show_suggestion_button"] = False
                prompt = "你能執行哪些與文本相關的操作，例如查找、解釋或摘要？"
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("思考中..."):
                    try:
                        answer, source_docs, chat_history = ask_question(st.session_state["rag_chain"], prompt)
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                        st.session_state["chat_history"].extend([{"type": "human", "content": prompt}, {"type": "ai", "content": answer}])
                        # 確保 current_sources 被正確儲存
                        st.session_state["current_sources"] = source_docs[:3] if source_docs else []
                        st.session_state["show_sources"] = True
                        # 保存對話紀錄
                        if st.session_state["current_chat_id"] is None:
                            new_chat_id = len(st.session_state["all_chat_history"])
                            st.session_state["current_chat_id"] = new_chat_id
                            st.session_state["all_chat_history"][new_chat_id] = {
                                "title": prompt[:10] if len(prompt) > 10 else prompt,
                                "history": list(st.session_state["chat_history"])
                            }
                        else:
                            st.session_state["all_chat_history"][st.session_state["current_chat_id"]]["history"] = list(st.session_state["chat_history"])
                            if not st.session_state["all_chat_history"][st.session_state["current_chat_id"]].get("title"):
                                st.session_state["all_chat_history"][st.session_state["current_chat_id"]]["title"] = prompt[:10] if len(prompt) > 10 else prompt
                    except Exception as e:
                        st.error(f"錯誤：處理問題時發生錯誤：{str(e)}")
                        answer = "抱歉，無法生成回答，請稍後再試。"
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                        st.session_state["chat_history"].extend([{"type": "human", "content": prompt}, {"type": "ai", "content": answer}])
                        st.session_state["current_sources"] = []
                    st.rerun()

        for idx, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # 如果是最後一條助理訊息，直接顯示參考文件（預設收合）
                if message["role"] == "assistant" and idx == len(st.session_state["messages"]) - 1:
                    if "current_sources" in st.session_state and st.session_state["current_sources"]:
                        with st.expander("參考文件"): # 移除 expanded=True，使其預設收合
                            for doc in st.session_state["current_sources"]:
                                st.markdown(f"{doc.page_content[:200]}...")
                                st.markdown("---")
                             
        # 聊天輸入框
        prompt = st.chat_input("輸入您的問題：", key="chat_input")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("思考中..."):
                try:
                    answer, source_docs, chat_history = ask_question(st.session_state["rag_chain"], prompt)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["chat_history"].extend([{"type": "human", "content": prompt}, {"type": "ai", "content": answer}])
                    # 確保 current_sources 被正確儲存
                    st.session_state["current_sources"] = source_docs[:3] if source_docs else []
                    st.session_state["show_sources"] = True
                    st.session_state["show_reference"] = False  # 重置顯示參考文件的狀態
                    # 保存對話紀錄
                    if st.session_state["current_chat_id"] is None:
                        new_chat_id = len(st.session_state["all_chat_history"])
                        st.session_state["current_chat_id"] = new_chat_id
                        st.session_state["all_chat_history"][new_chat_id] = {
                            "title": prompt[:10] if len(prompt) > 10 else prompt,
                            "history": list(st.session_state["chat_history"])
                        }
                    else:
                        st.session_state["all_chat_history"][st.session_state["current_chat_id"]]["history"] = list(st.session_state["chat_history"])
                        if not st.session_state["all_chat_history"][st.session_state["current_chat_id"]].get("title"):
                            st.session_state["all_chat_history"][st.session_state["current_chat_id"]]["title"] = prompt[:10] if len(prompt) > 10 else prompt
                except Exception as e:
                    st.error(f"錯誤：處理問題時發生錯誤：{str(e)}")
                    answer = "抱歉，無法生成回答，請稍後再試。"
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["chat_history"].extend([{"type": "human", "content": prompt}, {"type": "ai", "content": answer}])
                    st.session_state["current_sources"] = []
                st.rerun()

    else:
        st.error("⚠️ RAG 鏈初始化失敗，請檢查 API 金鑰和模型設定。")
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("輸入您的問題：", key="chat_input_api_error")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            st.warning("⚠️ API 金鑰未正確設定，可能無法進行問答。")

else:
    st.error("🔑 API 金鑰未找到或無效，請檢查您的設定。")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("輸入您的問題：", key="chat_input_no_api")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.warning("⚠️ API 金鑰未正確設定，可能無法進行問答。")