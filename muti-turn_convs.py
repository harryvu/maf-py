import asyncio
import os
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
thread = agent.get_new_thread()


def build_user_message(raw_input: str) -> ChatMessage:
    raw = raw_input.strip()

    if raw.lower().startswith("/image") or raw.lower().startswith("/img"):
        _, _, remainder = raw.partition(" ")
        remainder = remainder.strip()
        if not remainder:
            raise ValueError("Usage: /image <path-or-url> | <question>")

        if "|" in remainder:
            image_source, prompt = (part.strip() for part in remainder.split("|", 1))
        else:
            image_source, prompt = remainder, "Analyze this image."

        image_content = uri_content_from_image_source(image_source)

        return ChatMessage(
            role=Role.USER,
            contents=[
                TextContent(text=prompt),
                image_content,
            ],
        )

    return ChatMessage(role=Role.USER, contents=[TextContent(text=raw)])

async def main():
    print("Chat with the agent! Type 'quit' to exit.")
    print("To analyze an image: /image <path-or-url> | <question>\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            message = build_user_message(user_input)
        except Exception as exc:
            print(f"Input error: {exc}\n")
            continue

        result = await agent.run(message, thread=thread)
        print(f"Agent: {result.text}\n")

asyncio.run(main())

"""
Single agent with multiple conversations
----------------------------------------
It is possible to have multiple, independent conversations with the same agent instance, 
by creating multiple AgentThread objects. These threads can then be used to maintain 
separate conversation states for each conversation. The conversations will be fully 
independent of each other, since the agent does not maintain any state internally.
"""