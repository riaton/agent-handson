import streamlit as st


def create_state():
    """新しい状態を作成"""
    return {
        "containers": [],
        "current_status": None,
        "current_text": None,
        "final_response": ""
    }

def think(container, state):
    """思考開始を表示"""
    with container:
        thinking_status = st.empty()
        thinking_status.status("思考中", state="running")
    state["containers"].append((thinking_status, "思考中"))

def change_status(event, container, state):
    """サブエージェントのステータスを更新"""
    progress_info = event["subAgentProgress"]
    message = progress_info.get("message")
    stage = progress_info.get("stage", "processing")

    if state["current_status"]:
        status, old_message = state["current_status"]
        status.status(old_message, state="complete")

    with container:
        new_status_box = st.empty()
        if stage == "complete":
            display_state = "complete"
        else:
            display_state = "running"
        new_status_box.status(message, state=display_state)
    
    status_info = (new_status_box, message)
    state["containers"].append(status_info)
    state["current_status"] = status_info
    state["current_text"] = None
    state["final_response"] = ""


def stream(event, container, state):
    """テキストをストリーム表示"""
    delta = event["contentBlockDelta"]["delta"]
    if "text" not in delta:
        return
    
    if state["current_text"] is None:
        if state["containers"]:
            status, first_message = state["containers"][0]
            if "思考中" in first_message:
                status.status("思考中", state="complete")
        if state["current_status"]:
            status, message = state["current_status"]
            status.status(message, state="complete")

def finish(state):
    """表示の修了処理"""
    if state["current_text"]:
        response = state["final_response"]
        state["current_text"].markdown(response)
    for status, message in state["containers"]:
        status.status(message, state="complete")
        