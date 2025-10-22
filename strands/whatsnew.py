import os, feedparser, asyncio
import streamlit as st
from strands import Agent, tool


os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_REGION"] = st.secrets["AWS_REGION"]

@tool
def get_aws_updates(service_name: str) -> list:
    feed = feedparser.parse("https://aws.amazon.com/about-aws/whats-new/recent/feed/")
    results = []

    for entry in feed.entries:
        if service_name.lower() in entry.title.lower():
            results.append({
                "published": entry.get("published", "N/A"),
                "summary": entry.get("summary", "")
            })

            if len(results) >= 3:
                break

    return results

agent = Agent(
    model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools = [get_aws_updates] 
)

st.title("AWSアップデート確認くん")
service_name = st.text_input("アップデートを知りたいAWSサービス名を入力してください")

async def process_stream(service_name, container):
    text_holder = container.empty()
    response = ""
    prompt = f"AWSの{service_name.strip()}の最新のアップデートを、日付付きで要約して。"
    
    async for chunk in agent.stream_async(prompt):
        if not isinstance(chunk, dict):
            continue

        event = chunk.get("event")

        if isinstance(event, dict) and "contentBlockStart" in event:
            tool_use = event["contentBlockStart"].get("start", {}).get("toolUse", {})
            tool_name = tool_use.get("name")

            if response:
                text_holder.markdown(response)
                response = ""
            
            container.info(f"ツール実行中: {tool_name}")
            text_holder = container.empty()
        
        if text := chunk.get("data"):
            response += text
            text_holder.markdown(response)

if st.button("確認"):
    if service_name:
        with st.spinner("AWSアップデートを確認中..."):
            container = st.container()
            asyncio.run(process_stream(service_name, container))