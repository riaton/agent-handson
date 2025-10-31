import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_core import agent


def init_session_state():
    """セッション状態を初期化する"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'waiting_for_approval' not in st.session_state:
        st.session_state.waiting_for_approval = False
    if 'final_result' not in st.session_state:
        st.session_state.final_result = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = None

def reset_session():
    """セッション状態をリセットする"""
    st.session_state.messages = []
    st.session_state.waiting_for_approval = False
    st.session_state.final_result = None
    st.session_state.thread_id = None

#セッションの初期化を実行
init_session_state()

def run_agent(input_data):
    """エージェントを実行し、結果を処理する"""
    config = {
        "configurable": {"thread_id": st.session_state.thread_id}
    }
    with st.spinner("処理中...", show_time=True):
        for chunk in agent.stream(
            input_data,
            stream_mode="updates",
            config=config
        ):
            for task_name, result in chunk.items():
                if task_name == "__interrupt__":
                    st.session_state.tool_info = result[0].value
                    st.session_state.waiting_for_approval = True
                elif task_name == "agent":
                    st.session_state.final_result = result.content
                elif task_name == "invoke_llm":
                    if isinstance(chunk["invoke_llm"].content, list):
                        for content in result.content:
                            if content["type"] == "text":
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": content["text"]}
                                )
                elif task_name == "use_tool":
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "ツール実行!"}
                    )
        st.rerun()

def feedback():
    """フィードバックを取得し、エージェントに通知する関数"""
    approve_column, deny_column = st.columns(2)

    feedback_result = None
    with approve_column:
        if st.button("APPROVE", width="stretch"):
            st.session_state.waiting_for_approval = False
            feedback_result = "APPROVE"
    with deny_column:
        if st.button("DENY", width="stretch"):
            st.session_state.waiting_for_approval = False
            feedback_result = "DENY"

        #いずれかのボタンが押された場合
        return feedback_result

def app():
    st.title("Webリサーチエージェント")

    #メッセージ表示エリア
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # ツール承認の確認
    if st.session_state.waiting_for_approval and st.session_state.tool_info:
        st.info(st.session_state.tool_info["args"])
        if st.session_state.tool_info["name"] == "write_file":
            with st.container(height=400):
                st.html(st.session_state.tool_info["html"], width="stretch")
        feedback_result = feedback()
        if feedback_result:
            st.chat_message("user").write(feedback_result)
            st.session_state.messages.append(
                {"role": "user", "content": feedback_result}
            )
            run_agent(Command(resume=feedback_result))
            st.rerun()
    
    # ユーザ入力エリア
    if not st.session_state.waiting_for_approval:
        user_input = st.chat_input("メッセージを入力してください")
        if user_input:
            reset_session()
            st.session_state.thread_id = str(uuid.uuid4())
            st.chat_message("user").write(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            messages = [HumanMessage(content=user_input)]
            if run_agent(messages):
                st.rerun()
    else:
        st.info("ツールの承認待ちです。上記のボタンで応答してください")
    
if __name__ == "__main__":
    app()
