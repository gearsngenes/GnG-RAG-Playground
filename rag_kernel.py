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
import os

# === Custom module imports ===
from helpers import OPENAI_API_KEY, client as turbo_client, encode_image
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
#chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content="Hello"))
#chat_history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content="Hi there!"))

# === Define Custom Plugin Classes ===
class QueryPlugin:
    """
    Plugin for handling user queries, retrieving topic-relevant chunks,
    and answering using embedded data or fallback general knowledge (if permitted).
    """

    # @kernel_function(name="fetch_topic_descriptions",
    #                  description="Retrieve relevant topics and descriptions from Pinecone")
    # async def fetch_topic_descriptions(
    #         self,
    # ) -> Annotated[str, "A stringified dictionary of topics and their descriptions"]:
    #     topics_dict = vector_store_manager.get_descriptions()
    #     return str(topics_dict)

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
        #print("PRE: Found topics: ", response)
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

        # print("Found topics: ", topics)

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

        # Explicit logic for 'general'
        if found_list == ['general']:
            return general_knowledge_request if use_general_knowledge.lower() == "true" else no_information_found

        # Filter out 'general' if general knowledge not allowed
        if use_general_knowledge.lower() != "true":
            found_list = [topic for topic in found_list if topic.lower() != 'general']
            if not found_list:
                return no_information_found

        context_texts = []
        image_paths = []
        existing_indexes = vector_store_manager.list_indexes()

        for topic in found_list:
            if topic not in existing_indexes:
                continue
            metadata_list = vector_store_manager.query_at_index(topic, query)
            for metadata in metadata_list:
                chunk_type = metadata.get("type", "text")
                content = metadata.get("content", "")
                image_path = metadata.get("file_path")
                if chunk_type == "image":
                    image_paths.append(image_path)
                context_texts.append(f"{content}\n File source: {image_path}")

        if not context_texts and not image_paths:
            return general_knowledge_request if use_general_knowledge.lower() == "true" else no_information_found

        return str({"text_chunks": context_texts, "image_paths": image_paths})

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
                "in your uploaded documents to answer your question. Please try rephrasing or uploading new sources."
            )

        if retrieved_data == "No relevant context found":
            # Fall back to general knowledge
            prompt = f"Answer the following question using your general knowledge:\n\nQuery: {query}"
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
            response = await kernel.invoke_prompt(
                function_name="answer_query",
                plugin_name="QueryResponse",
                prompt=prompt,
                settings=settings,
            )
            return response

        # Standard case: we have valid retrieved content
        try:
            retrieved_dict = ast.literal_eval(retrieved_data)
        except Exception:
            return "⚠️ Error reading retrieved data format. Please retry."

        text_chunks = retrieved_dict.get("text_chunks", [])
        image_paths = retrieved_dict.get("image_paths", [])

        encoded_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                encoded_img = encode_image(img_path)
                encoded_images.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                )

        formatted_context = "\n\n".join(text_chunks)
        text_only_prompt = f"""
        Use the following retrieved information as context to intelligently
        answer the user's query. If the user asks for a source, please list
        the file path or image name associated with it. Be sure to ONLY use
        information that you can find from the retrieved chunks below to
        answer the query. Do NOT use your general knowledge base.
        
        Retrieved information:
        {formatted_context}
        
        Full conversation & User Query to answer:
        {query}
        """

        if encoded_images:
            full_prompt = text_only_prompt + "\n\n Please also use any retrieved images to aid in answering the query."
            messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}] + encoded_images}]
            response = turbo_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages
            )
            return response.choices[0].message.content
        else:
            settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
            response = await kernel.invoke_prompt(
                function_name="answer_query",
                plugin_name="QueryResponse",
                prompt=text_only_prompt,
                settings=settings,
            )
            return response

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
