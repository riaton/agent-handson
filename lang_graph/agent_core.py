from botocore.config import Config
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    ToolCall
)
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from dotenv import load_dotenv

# 環境変数ロード
load_dotenv(override=True)

web_search = TavilySearch(max_results=2, topic="general")
working_directory = "report"

file_toolkit = FileManagementToolkit(
    root_dir=str(working_directory),
    selected_tools=["write_file"]
)
write_file = file_toolkit.get_tools()[0]

tools = [web_search, write_file]
tools_by_name = {tool.name: tool for tool in tools}

cfg = Config(
    read_timeout=300
)

llm_with_tools = init_chat_model(
    model="us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse",
    config=cfg
).bind_tools(tools)

system_prompt = """
あなたの責務はユーザーからのリクエストを調査し、調査結果をファイル出力することです
- ユーザーのリクエスト調査にWeb検索が必要であれば、web検索ツールを使ってください
- 必要な情報が集まったと判断したら検索は終了してください
- 検索は最大2回までとしてください
- ファイル出力はHTML形式(.html)に変換して保存してください
  - Web検索が拒否された場合、Web検索を中止してください
  - レポート保存を拒否された場合、レポート作成を中止し、内容をユーザに直接伝えてください
"""

# LLMを呼び出すタスク
@task
def invoke_llm(messages: list[BaseMessage]) -> AIMessage:
    response = llm_with_tools.invoke(
        SystemMessage(content=system_prompt) + messages
    )
    return response

# ツールを実行するタスク
@task
def use_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])

# ユーザーにツール実行の承認を求める
def ask_human(tool_call: ToolCall):
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_data = {"name": tool_name}
    if tool_name == web_search.name:
        args = f'* ツール名\n'
        args += f'  * {tool_name}\n'
        args += f'* 引数\n'
        for key, value in tool_args.items():
            args += f'  *{key}\n'
            args += f'  * {value}\n'
        tool_data["args"] = args
    elif tool_name == write_file.name:
        args = f'* ツール名\n'
        args += f'  * {tool_name}\n'
        args += f'* 保存ファイル名\n'
        args += f'  * {tool_args["file_path"]}'
        tool_data["html"] = tool_args["text"]
    tool_data["args"] = args

    feedback = interrupt(tool_data)

    if feedback == "APPROVE":
        return tool_call
    
    return ToolMessage(
        content="ツール利用が拒否されたため、処理を終了してください",
        name=tool_name,
        tool_call_id=tool_call["id"]
    )

checkpointer = MemorySaver()
@entrypoint(checkpointer)
def agent(messages):
    llm_response = invoke_llm(messages).result()

    while True:
        if not llm_response.tool_calls:
            break

        approved_tools = []
        tool_results = []

        for tool_call in llm_response.tool_calls:
            feedback = ask_human(tool_call)
            if isinstance(feedback, ToolMessage):
                tool_results.append(feedback)
            else:
                approved_tools.append(feedback)

        tool_futures = []
        for tool_call in approved_tools:
            future = use_tool(tool_call)
            tool_futures.append(future)

        tool_use_resutls = []
        for future in tool_futures:
            result = future.result()
            tool_use_resutls.append(result)

        messages = add_messages(messages, [llm_response, *tool_use_resutls, *tool_results])

        llm_response = invoke_llm(messages).result()
        
    return llm_response