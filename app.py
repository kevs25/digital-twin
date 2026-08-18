import os
from context import TWIN_SYSTEM_PROMPT
from tools import record_user_details, record_unknown_question
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
from agents import Agent, Runner

load_dotenv(override=True)

MODEL_NAME = "gpt-5.4-mini"

agent = Agent(
    name="Digital Twin",
    model=MODEL_NAME,
    instructions=TWIN_SYSTEM_PROMPT,
    tools=[record_user_details, record_unknown_question],
)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


async def chat(message, history):
    clean_history = [{"role": item["role"], "content": extract_text(item["content"])} for item in history]
    input_items = clean_history + [{"role": "user", "content": message}]
    result = await Runner.run(agent, input_items)
    return result.final_output


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(
        css=CSS,
        js=JS,
        theme=gr.themes.Base(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
