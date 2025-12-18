import asyncio
import os
import sys
from dotenv import load_dotenv
from agent_framework.azure import AzureOpenAIChatClient

from agent_framework import ChatMessage, Role, TextContent

from image_utils import uri_content_from_image_source

load_dotenv()

agent = AzureOpenAIChatClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
).create_agent(
    name="VisionAgent",
    instructions="You are a helpful agent that can analyze images"
)


def build_message_from_cli() -> ChatMessage:
    # Usage:
    #   python vision_agent.py <image-path-or-url> [prompt...]
    # If omitted, we fall back to an interactive prompt.
    if len(sys.argv) >= 2:
        image_source = sys.argv[1]
        prompt = " ".join(sys.argv[2:]).strip() or "Analyze this image."
        return ChatMessage(
            role=Role.USER,
            contents=[
                TextContent(text=prompt),
                uri_content_from_image_source(image_source),
            ],
        )

    print("Enter an image path or URL (or 'quit'): ")
    while True:
        image_source = input("> ").strip()
        if image_source.lower() == "quit":
            raise SystemExit(0)
        if image_source:
            break

    prompt = input("Question (blank = analyze): ").strip() or "Analyze this image."
    return ChatMessage(
        role=Role.USER,
        contents=[
            TextContent(text=prompt),
            uri_content_from_image_source(image_source),
        ],
    )

async def main():
    message = build_message_from_cli()
    result = await agent.run(message)
    print(result.text)

if __name__ == "__main__":
    asyncio.run(main())