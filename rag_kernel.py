import asyncio
import ast
import os
from typing import Annotated
from urllib.parse import quote

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.kernel import Kernel
from semantic_kernel.planners.sequential_planner import SequentialPlanner
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

from helpers import OPENAI_API_KEY, encode_image, UPLOAD_FOLDER
from qdrant_utils import vector_store_manager

# === Kernel Initialization ===
kernel = Kernel()
service_id = "chat"
model_id = "gpt-4o"
kernel.add_service(OpenAIChatCompletion(service_id=service_id, api_key=OPENAI_API_KEY, ai_model_id=model_id))
planner = SequentialPlanner(kernel, service_id=service_id)

settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["QueryResponse"]})
chat_history = ChatHistory()

def format_history() -> str:
    """
    Formats the Semantic Kernel chat history (ChatMessageContent entries) into a readable string transcript.

    Returns:
        str: Plain-text conversation log with roles and content.
    """
    return "\n".join(
        f"{msg.role.value.capitalize()}: {msg.content.strip()}"
        for msg in chat_history.messages
        if msg.content and msg.content.strip()
    )


# === Query Plugin ===
class QueryPlugin:

    @kernel_function(name="retrieve_chunks",
                     description="Retrieve relevant text/image chunks from selected topics with source markdown formatting.")
    async def retrieve_chunks(self,
                              kernel: Kernel,
                              query: Annotated[str, "The user's question."],
                              topics: Annotated[str, "A stringified list of selected topic names."]) -> str:
        topics = ast.literal_eval(topics)
        if "general" in topics:
            return "general_knowledge_only"

        context_texts = []
        image_descriptions = []

        for topic in topics:
            if topic not in vector_store_manager.list_indexes():
                continue

            metadata_list = vector_store_manager.query_at_index(topic, query)

            for metadata in metadata_list:
                content = metadata.get("content", "")
                file_path = metadata.get("file_path", "").replace("\\", "/")
                filename = os.path.basename(file_path)
                rel_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
                url = f"/{UPLOAD_FOLDER}/{quote(rel_path)}"

                if metadata.get("type") == "image":
                    image_descriptions.append(f"![{filename}]({url})\nDescription: {content}")
                else:
                    context_texts.append(f"{content}\nSource URL: [{filename}]({url})")

        if not context_texts and not image_descriptions:
            return "general_knowledge_only"

        formatted = "<TEXT CHUNKS>\n" + "\n\n".join(context_texts)
        if image_descriptions:
            formatted += "\n\n<IMAGE DESCRIPTIONS>\n" + "\n\n".join(image_descriptions)

        return formatted

    @kernel_function(name="generate_answer",
                     description="Answer the user query with embedded citations and markdown output.")
    async def generate_answer(self,
                              kernel: Kernel,
                              query: Annotated[str, "The user's question."],
                              context: Annotated[str, "Retrieved context chunks or 'general_knowledge_only'."]) -> str:
        if context == "general_knowledge_only":
            prompt = f"Answer using general knowledge:\n\n{query}"
        else:
            from prompts import full_prompt_4o
            prompt = full_prompt_4o(context, format_history(), query)
        return await kernel.invoke_prompt(
            function_name="generate_answer",
            plugin_name="QueryResponse",
            prompt=prompt,
            settings=kernel.get_prompt_execution_settings_from_service_id(service_id)
        )

kernel.add_plugin(QueryPlugin(), "QueryResponse")
kernel.add_plugin(TextPlugin(), plugin_name="text")

# === Run Query with Planner ===
async def run_query_pipeline(query: str, topics: list[str]) -> dict:
    chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=query))

    goal_prompt = f"""
Use the selected topics to retrieve relevant context (including text and image info),
then generate a <think>/<response> markdown-formatted answer that embeds sources and explains your reasoning.
"""

    plan = await planner.create_plan(goal_prompt)
    execution_result = await plan.invoke(kernel, {
        "query": query,
        "topics": str(topics)
    })

    output = str(execution_result.value)
    think_start, think_end = output.find('<think>'), output.find('</think>')
    response_start, response_end = output.find('<response>'), output.find('</response>')

    think = output[think_start+7:think_end].strip() if think_start != -1 and think_end != -1 else ""
    response = output[response_start+10:response_end].strip() if response_start != -1 and response_end != -1 else output.strip()

    chat_history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=response))
    return {"think": think, "response": response}

def run_query(query: str, topics: list[str]) -> dict:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_query_pipeline(query, topics))

def clear_sk_memory():
    chat_history.clear()

def get_sk_chat_history():
    return [{"role": msg.role.value, "content": msg.content} for msg in chat_history.messages]
