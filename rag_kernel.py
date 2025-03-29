# === Base imports ===
import asyncio
from typing import Annotated
import ast
import re
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.kernel import Kernel
from semantic_kernel.planners.sequential_planner import SequentialPlanner
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from urllib.parse import quote
import os

# === Custom module imports ===
from helpers import OPENAI_API_KEY, client as turbo_client, encode_image, UPLOAD_FOLDER
from pinecone_utils import vector_store_manager

# === Initialize Kernel & Planner Globally ===
"""
The kernel acts as the central hub of various services, planners, and
plugins, which can then be invoked or used at various times.
"""
kernel = Kernel()
service_id = "chat"
ai_model_id = "gpt-4o"
ai_service = OpenAIChatCompletion(service_id=service_id, api_key=OPENAI_API_KEY, ai_model_id=ai_model_id)
kernel.add_service(ai_service)

# === Initialize the Planner ===
"""
Planners are what we can use to intelligently construct sequences of method calls
based on a user's goal. Right now, we're using a sequential planner, but others
exist. This will be how we handle sending our query through a sequence of methods
without explicitly defining what that sequence is.
"""
planner = SequentialPlanner(kernel, service_id=service_id)
settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["QueryResponse"]})

# === Initialize Chat History ===
chat_history = ChatHistory()

# === Define Custom Plugin Classes ===
class QueryPlugin:
    """
    Plugin for handling user queries, retrieving topic-relevant chunks,
    and answering using embedded data or fallback general knowledge (if permitted).
    """

    @kernel_function(name="determine_relevant_topics",
                     description="Identify the most relevant topics for the user's query")
    async def determine_relevant_topics(
            self,
            kernel: Kernel,
            query: Annotated[str, "The conversation and user's latest query"],
            topics: Annotated[str, "A stringified list of topics that was supplied by the user directly. This can either be empty, signaling us to choose the topics most related to the query ourselves, or it could contain a preselected list of topics to use."],
            # topics_descriptions: Annotated[str, "A stringified dictionary of topics and descriptions"]
    ) -> Annotated[str, "Stringified list of most relevant topic names, or 'general' if none are applicable"]:
        topics = ast.literal_eval(topics)
        if len(topics):
            return str(topics)

        # topics_dict = ast.literal_eval(topics_descriptions)
        prompt = f"""
        Given the following topics and their descriptions:
        
        {vector_store_manager.get_descriptions()}
        
        compare the content of the query with the descriptions
        of the topics and select the ones most relevant to
        answering the query: "{query}".
        
        Be sure to ONLY return a list formatted like: ['Topic1', 'Topic2']".
        Do NOT add any trailing whitespaces, extra quotation marks,
        or 'python' tags or anything like that
        If none are applicable, return ['general'].
        """

        settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
        response = await kernel.invoke_prompt(
            function_name="determine_relevant_topics",
            plugin_name="QueryResponse",
            prompt=prompt,
            settings=settings,
        )
        return response

    @kernel_function(name="retrieve_context_chunks",
                     description="Retrieve relevant chunks from Pinecone indices, including images.")
    async def retrieve_context_chunks(
            self,
            kernel: Kernel,
            found_topics: Annotated[str, "Stringified list of relevant topic names that have been selected to best answer the query"],
            query: Annotated[str, "The user query"],
            use_general_knowledge: Annotated[str, "Whether to fall back to general knowledge"]
    ) -> Annotated[str, "Stringified list of retrieved text/image descriptions and image paths"]:

        general_knowledge_request = "No relevant context found"
        no_information_found = "no_information_found"

        def find_valid_list(text):
            match = re.search(r"\[\s*(?:'[^']*'(?:\s*,\s*'[^']*')*)?\s*\]", text)
            return match.group(0) if match else None

        found_list_str = find_valid_list(found_topics)
        if not found_list_str:
            return general_knowledge_request if use_general_knowledge.lower() == "true" else no_information_found

        try:
            found_list = ast.literal_eval(found_list_str)
        except Exception:
            return no_information_found

        if found_list == ['general']:
            return general_knowledge_request if use_general_knowledge.lower() == "true" else no_information_found

        if use_general_knowledge.lower() != "true":
            found_list = [topic for topic in found_list if topic.lower() != 'general']
            if not found_list:
                return no_information_found

        context_texts = []
        image_paths = []
        file_links = []
        existing_indexes = vector_store_manager.list_indexes()

        for topic in found_list:
            if topic not in existing_indexes:
                continue
            metadata_list = vector_store_manager.query_at_index(topic, query)
            for metadata in metadata_list:
                chunk_type = metadata.get("type", "text")
                content = metadata.get("content", "")
                file_path = metadata.get("file_path").replace("\\", "/")

                if file_path.startswith(f"{UPLOAD_FOLDER}/"):
                    relative_path = file_path[len(f"{UPLOAD_FOLDER}/"):]
                else:
                    relative_path = file_path
                url_path = f"/{UPLOAD_FOLDER}/{quote(relative_path)}"
                filename = os.path.basename(file_path)
                markdown_link = f"[{filename}]({url_path})"

                if chunk_type == "image":
                    image_paths.append(file_path)

                context_texts.append(f"{content}\nSource URL: {markdown_link}")
                file_links.append(markdown_link)

        if not context_texts and not image_paths:
            return general_knowledge_request if use_general_knowledge.lower() == "true" else no_information_found

        return str({"text_chunks": context_texts, "image_paths": image_paths, "file_links": file_links})

    @kernel_function(name="answer_query",
                     description="Answer the user query with retrieved context, including images if available.")
    async def answer_query(
            self,
            query: Annotated[str, "The user query"],
            retrieved_data: Annotated[str, "Stringified dictionary containing relevant text_chunks and image_paths"]
    ) -> Annotated[str, "Final answer to the user query"]:

        if retrieved_data == "no_information_found":
            return (
                "❌ Sorry, we couldn’t find any relevant topics or matching content "
                "in your uploaded documents to answer your question. Please try rephrasing "
                "your query or uploading new sources."
            )

        if retrieved_data == "No relevant context found":
            prompt = f"Answer the following question using your general knowledge:\n\nQuery: {query}"
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
            response = await kernel.invoke_prompt(
                function_name="answer_query",
                plugin_name="QueryResponse",
                prompt=prompt,
                settings=settings,
            )
            return response

        try:
            retrieved_dict = ast.literal_eval(retrieved_data)
        except Exception:
            return "⚠️ Error reading retrieved data format. Please retry."

        text_chunks = retrieved_dict.get("text_chunks", [])
        image_paths = retrieved_dict.get("image_paths", [])
        file_links = list(set(retrieved_dict.get("file_links", [])))

        encoded_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                encoded_img = encode_image(img_path)
                ext = os.path.splitext(img_path)[1].lower().lstrip(".")
                encoded_images.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{encoded_img}"}}
                )

        formatted_context = "\n\n".join(text_chunks)
        prompt = f"""
    You are a precise assistant generating Markdown-ready answers for a web app.
    
    Use the retrieved information below to answer the user's query. **Only** use
    information from these chunks. Do **not** rely on general knowledge. If you
    want to cite a specific source, use the corresponding Markdown link provided
    next to each chunk to indicate the information source when inserting it into
    the response.
    
    Return ONLY the response itself.
    ---
    
    Retrieved Chunks:
    {formatted_context}
    
    ---
    User Query:
    {query}
    """

        if encoded_images:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + encoded_images}]
            raw_response = turbo_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            ).choices[0].message.content
        else:
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
            raw_response = await kernel.invoke_prompt(
                function_name="answer_query",
                plugin_name="QueryResponse",
                prompt=prompt,
                settings=settings,
            )

        final_answer = str(raw_response).strip()
        # all_links = "\n".join(file_links)
#         final_response = f"""
# {final_answer}
#
# ### Sources
#
# {all_links}
# """

        return final_answer#final_response

# === Add Plugins to the Kernel ===
kernel.add_plugin(QueryPlugin(), plugin_name="QueryResponse",
                  description="""
                  For question-answering related functions 
                  for identifying and selecting relevant 
                  topics for answering a query, retrieval of 
                  relevant content for context based on
                  those selected topics, answering 
                  user queries, and formatting the responses"""
                  )
kernel.add_plugin(TextPlugin(), plugin_name="text")


async def run_query_pipeline(user_query: str, topics: list[str], use_general_knowledge: bool):
    history_text = "\n".join(
        f"{msg.role.value}: {msg.content}" for msg in chat_history.messages
    )
    full_prompt = f"""
    This is the prior messages exchanged in a conversation:

    {history_text}

    Use this chat history to answer the user's newest query.
    
    User Query: {user_query}
    """
    chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=user_query))
    goal_prompt = f"""
    Ingest the prior conversation and the current user query,
    then -if a list of topics haven't been provided by the user-,
    select a topic that best fits what the user's query is asking,
    retrieve information from those topics that is most
    relevant to the user's query (including text and
    potentially images), and finally answer the said query
    using the retrieved information as context. Then make sure
    the final response is in proper markdown format and
    cleaned up for display.
    """
    plan = await planner.create_plan(goal_prompt)
    execution_result = await plan.invoke(kernel, {
        "query": full_prompt,
        "topics": str(topics),
        "use_general_knowledge": str(use_general_knowledge)
    })
    chat_history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=execution_result.value))
    return execution_result.value

def run_query(user_query: str, topics: list[str], use_general_knowledge: bool = True):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(run_query_pipeline(user_query, topics, use_general_knowledge))
    return response

def clear_sk_memory():
    """Clears the stored conversation history in SK's built-in text memory."""
    chat_history.clear()

def get_chat_history():
    """Returns the chat history as a list of dictionaries with role and content."""
    return [
        {"role": msg.role.value, "content": msg.content}
        for msg in chat_history.messages
    ]


"""
For running the method locally to test out semantic kernel
responses locally. Must already have data in your pinecone
database to use the below lines correctly.
"""
# if __name__ == "__main__":
#      user_query = input("User query: ")
#      print(asyncio.run(run_query_pipeline(user_query)))
#      clear_sk_memory()
