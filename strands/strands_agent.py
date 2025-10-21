from strands import Agent
from dotenv import load_dotenv

load_dotenv()

agent = Agent("us.anthropic.claude-sonnet-4-20250514-v1:0")
agent("Strandsってどう言う意味？")