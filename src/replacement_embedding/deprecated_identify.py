import csv
import re
import os
import csv
from openai import OpenAI
from markdown import markdown
from bs4 import BeautifulSoup
import sys
from tqdm import tqdm
import json 

csv.field_size_limit(50_000_000)

API_TOKEN = ""

ds_client = OpenAI(
	base_url="https://api.deepseek.com/",
	api_key=API_TOKEN,
)

def get_ds_content(prompt,temperature=0.7,role_system="You are a helpful assistant"): #not stream
    response = ds_client.chat.completions.create(
        model="deepseek-chat",
	    stream=False,
	    temperature=0.7,
	    messages=[
		{
			"role": "system",
			"content": role_system
		},
		{
			"role": "user",
			"content": prompt
		}
	    ],
    )
    content = response.choices[0].message.content
    return content

def clean_text(text: str) -> str:
    """
    文本预处理：
    1. 去除所有URL
    2. 转换Markdown为纯文本
    3. 去除多余空白字符
    """
    if not text:
        return ""

    # 去除URL（保留被 [] 包裹的文本）
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Markdown转HTML再提取纯文本
    html = markdown(text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")

    # 可选：其他清洗规则
    text = re.sub(r'\s+', ' ', text).strip()  # 合并连续空白
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # 去除方括号但保留内容

    return text

def deprecation_identification(readme_text):
    readme_text = clean_text(readme_text)
    prompt = f'''You are an assistant that extracts replacement package information from technical documentation.

            Given a paragraph, determine whether it indicates that a package is deprecated or replaced. If it does, extract and return the name of the recommended replacement package.

            Respond with:
            - "None" if there is no replacement package mentioned.
            - The exact name of the recommended replacement package if one is clearly stated.

            Only return the package name or "None".

            Paragraph:
            """
            {readme_text}
            """
            '''
    content = get_ds_content(prompt)
    return content

# 常见的表示“已弃用”的关键词
DEPRECATED_KEYWORDS = [
    "deprecated",
    "deprecation",
    "deprecate",
    "DeprecationWarning",
    "no longer maintained",
    "no longer supported",
    "no longer available",
    "no longer in use",
    "unmaintained",
    "has been deprecated",
    "this crate is deprecated",
    "this project is deprecated",
    "this library is deprecated",
    "replaced by",
    "superseded by",
    "moved to",
    "use instead",
    "recommend using",
    "recommend to use",
    "we suggest using",
    "maintenance mode",
    "archived",
    "abandoned",
    "no active development",
    "will not receive updates",
    "use alternative",
    "consider switching to",
]

RE_DEPRECATED_PATTERNS = [
    r'\bdeprecated\b',
    r'\bdeprecation\b',
    r'\bDeprecationWarning\b',
    r'\bno longer (maintained|supported|available|in use)\b',
    r'\b(un)?maintained\b',
    r'\bhas been deprecated\b',
    r'\breplaced (by|with)\b',
    r'\bsuperseded (by|with)\b',
    r'\bmoved to\b',
    r'\brecommend(ed)? (to use|using)\b',
    r'\bsuggest(ed)? (to use|using)\b',
    r'\bmaintenance mode\b',
    r'\barchived\b',
    r'\babandoned\b',
    r'\bno active development\b',
    r'\buse alternative\b',
    r'\bconsider switching to\b',
    r'\bwill not receive updates\b',
]

def contains_deprecated_message(text):
    if not text:
        return False
    text = text.lower()
    return any(re.search(pattern, text) for pattern in RE_DEPRECATED_PATTERNS)

def extract_deprecated_snippet(text, window=5):
    if not text:
        return None
    text = text.strip().replace("\n", " ")
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)

    for i, sentence in enumerate(sentences):
        for pattern in RE_DEPRECATED_PATTERNS:
            if re.search(pattern, sentence.lower()):
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                snippet = " ".join(sentences[start:end])
                return snippet.strip()
    return None


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_row(row, all_pkg_names):
    pkg_name = row.get("name")
    for field_name, field_value in row.items():
        txt = extract_deprecated_snippet(str(field_value))
        if txt is not None:
            replace_pkg = deprecation_identification(txt)
            if replace_pkg != 'None':
                if replace_pkg.strip().lower() in all_pkg_names and replace_pkg != pkg_name:
                    return {'name': pkg_name, 'replacement': replace_pkg, 'message': txt}
    return None

def check_csv_for_deprecated_messages(csv_file_path):
    all_pkg_names = set()
    with open(os.path.join(csv_file_path, 'crates.csv'), newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row.get("name")
            if name:
                all_pkg_names.add(name.strip().lower())

    with open(os.path.join(csv_file_path, 'crates.csv'), newline='', encoding='utf-8') as csvfile:
        rows = list(csv.DictReader(csvfile))

    result = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, row, all_pkg_names): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                res = future.result()
                if res:
                    # print(f"  替换元组: ({res['name']}, {res['replacement']})")
                    result.append(res)
                    with open(os.path.join(csv_file_path, 'deprecated-large.json'), "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error processing row: {e}")


# 用法示例
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "..", "data")
check_csv_for_deprecated_messages(csv_path)
