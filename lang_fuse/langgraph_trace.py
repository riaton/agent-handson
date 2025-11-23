from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv
load_dotenv()

# Web検索ツールの初期化
web_search = TavilySearch(max_results=3, topic="general")

# ReactAgentの構築
tools = [web_search]
# モデルの準備
llm = init_chat_model(
    model="us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse",
)
# ReActエージェントの構築
agent = create_react_agent(llm, tools)

# ReActエージェントの実行
langfuse_handler = CallbackHandler()
messages = agent.invoke(
    {
        "messages": [
            ("human", "AIエージェントの最新動向を教えてください。検索は一度だけ実行してください。")
        ]
    },
    config={"callbacks": [langfuse_handler]},
)

for message in messages["messages"]:
    message.pretty_print()