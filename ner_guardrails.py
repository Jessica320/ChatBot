import re
from transformers import BertTokenizerFast, AutoModelForTokenClassification
from transformers import pipeline

# 定義本地模型路徑
# NER模型
local_model_path = "./bert-base-chinese-ner"

# 載入模型和 tokenizer
try:
    tokenizer = BertTokenizerFast.from_pretrained(local_model_path)
    model = AutoModelForTokenClassification.from_pretrained(local_model_path)
    # 創建 NER 管道 (這部分在模組載入時執行一次)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
except Exception as e:
    print(f"載入模型或 tokenizer 時發生錯誤: {e}")
    ner_pipeline = None

def extract_entities_with_regex(text):
    """
    使用 NER 模型和正則表達式提取文本中的實體 (包含組織、電話號碼、電子郵件)。
    Args:
        text (str): 需要提取實體的文本。
    Returns:
        list: 包含識別出的實體列表，每個實體是一個字典。
    """
    ner_results = []
    if ner_pipeline:
        ner_results.extend(ner_pipeline(text))

    # 使用正則表達式尋找電話號碼
    phone_pattern = r"(?:\+?886-?|0)?9\d{2}-?\d{3}-?\d{3}|(?:\+?886-?|0)?\d{2}-?\d{4}-?\d{4}|\d{2}-\d{7,8}|\d{4}-\d{7}"
    phones = re.findall(phone_pattern, text)
    for phone in phones:
        start_index = text.find(phone)
        end_index = start_index + len(phone)
        ner_results.append({
            'entity': 'PHONE',
            'score': 0.95,
            'index': -1,
            'word': phone,
            'start': start_index,
            'end': end_index,
            'entity_group': 'PHONE'
        })

    # 使用正則表達式尋找電子郵件地址
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    for email in emails:
        start_index = text.find(email)
        end_index = start_index + len(email)
        ner_results.append({
            'entity': 'EMAIL',
            'score': 0.99,
            'index': -1,
            'word': email,
            'start': start_index,
            'end': end_index,
            'entity_group': 'EMAIL'
        })

    return ner_results

def desensitize_text_with_entities(text, ner_results):
    """
    使用 NER 結果去敏化文本中的個人敏感資訊 (包含人名、組織、電話號碼、電子郵件、民族/宗教/政治團體)。
    Args:
        text (str): 需要去敏化的原始文本。
        ner_results (list): NER 模型識別出的實體列表。
    Returns:
        str: 去敏化後的文本，敏感資訊已被 '[REDACTED]' 替換。
    """
    desensitized_text = text
    # 從後向前替換，防止位置偏移
    for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
        if entity['entity_group'] in ['PERSON', 'ORG', 'PHONE', 'EMAIL', 'NORP']:
            desensitized_text = desensitized_text[:entity['start']] + '[REDACTED]' + desensitized_text[entity['end']:]
    return desensitized_text

if __name__ == "__main__":
    # 範例使用
    text_to_process = "凱基證券(phone：0223148800)資訊部資料科學家許大明，他的email是example@gmail.com。他是個無神教"
    entities = extract_entities_with_regex(text_to_process)
    #print("識別出的實體:", entities)
    desensitized_text = desensitize_text_with_entities(text_to_process, entities)
    print("去敏化後的文本:", desensitized_text)

