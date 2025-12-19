import os

from dotenv import load_dotenv
from google import genai


# Load environment variables from a local .env file (if present).
# Note: `.env` is NOT automatically loaded by Python or the GenAI SDK.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
    )

client = genai.Client(api_key=api_key)

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works for a software developer."
# )
# print(response.text)

stream = client.interactions.create(
    model="gemini-2.5-flash",
    input="Explain quantum entanglement for a physicist.",
    stream=True
)

for chunk in stream:
    if chunk.event_type == "content.delta":
        if chunk.delta.type == "text":
            print(chunk.delta.text, end="", flush=True)
        elif chunk.delta.type == "thought":
            print(chunk.delta.thought, end="", flush=True)
    elif chunk.event_type == "interaction.complete":
        print(f"\n\n--- Stream Finished ---")
        print(f"Total Tokens: {chunk.interaction.usage.total_tokens}")