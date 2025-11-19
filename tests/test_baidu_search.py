import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('BAIDU_SEARCH_API_KEY')
QUERY = '中关村三小足球赛'
URL = 'https://qianfan.baidubce.com/v2/ai_search/web_search'
# 请求体
payload = {
    "messages": [
        {
            "content": QUERY,
            "role": "user"
        }
    ],
    "search_source": "baidu_search_v2",
    "resource_type_filter": [{"type": "web", "top_k": 20}],
    "search_recency_filter": "year"
}
# 请求头
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
# 发起POST请求
response = requests.post(URL, json=payload, headers=headers)
# 解析响应
if response.status_code == 200:
    data = response.json()
    references = data.get('references') or data.get(
        'result', {}).get('items') or []

    for item in references:
        print(f"标题: {item.get('title')}")
        print(f"链接: {item.get('url')}")
        print(f"内容: {item.get('content')}")  # 获取具体内容
        print("-" * 50)
else:
    print(f"请求失败: {response.status_code}, 错误信息: {response.text}")
