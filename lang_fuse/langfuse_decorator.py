import boto3
import os
from langfuse import observe
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()

bedrock_client = boto3.client("bedrock-runtime")
model_id = "us.amazon.nova-premier-v1:0"

# Web検索クエリ
@observe
def create_query(query: str):
    system_prompt = """ユーザーからの問い合わせ内容をWeb検索し、レポートを作成します。
    Web検索用のクエリを1つ作成してください。検索単語以外は回答しないでください。"""
    prompt = f"ユーザーの質問: {query}"

    system = [
        { "text": system_prompt }
    ]
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system,
    )

    return response["output"]["message"]["content"][0]["text"]

# Tavilyを使ったWeb検索を実行する関数
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@observe
def web_search(query: str):
    """Get content related the query from web."""
    search_result = tavily_client.search(
        query=query,
        num_results=3,
    )

    return [doc["content"] for doc in search_result["results"]]

@observe
def create_report(query: str, contents: list[str]):
    system_prompt = """Web検索した結果とユーザークエリを元にMarkdownのレポートを
    作成してください。タイトルと見出しも作成してください。"""

    prompt = f"ユーザーの質問: {query}\n\n web検索結果: {"\n".join(contents)}"
    system = [
        { "text": system_prompt }
    ]
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system,
    )

    return response["output"]["message"]["content"][0]["text"]

# 各タスクを呼び出す関数
@observe
def workflow(query: str):
    web_query = create_query(query)
    contents = web_search(web_query)
    report = create_report(query, contents)

    return report

query = "LangChainとLangGraphのユースケースの違いについて教えてください。"
report = workflow(query)

print(report)
