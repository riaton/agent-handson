import feedparser
from strands import Agent, tool
from dotenv import load_dotenv

load_dotenv()

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