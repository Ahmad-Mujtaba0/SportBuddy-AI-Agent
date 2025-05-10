from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
set_default_openai_client(provider)
set_tracing_disabled(True)

Model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

def agent_run():
    sportbuddy = Agent(
        name="SportBuddy",
        instructions="""
        You are SportBuddy, an enthusiastic and motivating assistant for sports and fitness enthusiasts. 
        Your role is to inspire users to stay active, provide general tips for improving their sports performance, and encourage a positive mindset toward physical activity. 
        Be upbeat, supportive, and inclusive, catering to all skill levels without pushing overly intense routines. 
        Use energetic, friendly language like a passionate coach or teammate. 
        Never offer medical or injury-related advice. 
        Always end your response with an encouraging question to keep the user motivated.
        """,
        model=Model
    )

    result = Runner.run_sync(sportbuddy, "What are some fun ways to improve my stamina for soccer?")
    
    artifact_id = str(uuid.uuid4())
    filename = "README.md"

    markdown_content = f"""\n\n---\n\n### **Artifact ID:** `{artifact_id}`

**Question:** *What are some fun ways to improve my stamina for soccer?*

**SportBuddy says:**
{result}
"""

    with open(filename, "a", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Response appended to {filename}")
