import asyncio
import operator
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing import Annotated, Dict, List, Union
from dotenv import load_dotenv

# 環境変数ロード
load_dotenv()

mcp_client = None
tools = None
llm_with_tools = None

async def init_llm():
    """MCPクライアントとツールを初期化する"""
    global mcp_client, tools, llm_with_tools

    mcp_client = MultiServerMCPClient(
        {
            # File System MCP Server
            "file-system": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"],
                "transport": "stdio",
            },
            # AWS Knowledge MCP Server
            "aws-knowledge": {
                "url": "https://knowledge-mcp.global.api.aws",
                "transport": "streamable_http",
            }
        }
    )

    tools = await mcp_client.get_tools()
    llm_with_tools = init_chat_model(
        model="us.amazon.nova-premier-v1:0",
        model_provider="bedrock_converse"
    ).bind_tools(tools)

# ステートの定義
class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add]

system_prompt = """
あなたの責務はAWSドキュメントを検索し、Markdown形式でファイル出力することです。
- 検索後、Markdown形式に変換してください。
- 検索は最大で2回までとし、その時点での情報を出力してください。
"""

async def agent(state: AgentState) -> Dict[str, List[AIMessage]]:
    response = await llm_with_tools.ainvoke(
        [SystemMessage(system_prompt)] + state.messages
    )

    return {"messages": [response]}

def route_node(state: AgentState) -> Union[str]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("[AI Message]以外のメッセージです。遷移が不正な可能性があります。")
    if not last_message.tool_calls:
        return END
    return "tools"

async def main():
    await init_llm()
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_node)
    builder.add_edge("tools", "agent")
    graph = builder.compile(name="ReAct Agent")

    question = "Bedrockで利用可能なモデルプロバイダーを教えて"
    response = await graph.ainvoke(
        {"messages": [HumanMessage(question)]}
    )
    print(response)
    return response

asyncio.run(main())