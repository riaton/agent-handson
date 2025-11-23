from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv
load_dotenv()

# モデルの準備
llm = init_chat_model(
    model="us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse",
)

langfuse_handler = CallbackHandler()
config = {"callbacks": [langfuse_handler]}

response = llm.invoke("こんにちは", config=config)
print(response)