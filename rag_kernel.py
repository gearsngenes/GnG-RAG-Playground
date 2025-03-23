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
    This plugin focuses on taking in user queries and using text & images
    to provide relevant answers. It focuses currently only on handling
    text-to-text responses, though it can use images internally for more
    context to answer.
    """

    @kernel_function(name="fetch_topic_descriptions",
                     description="Retrieve relevant topics and descriptions from Pinecone")
    async def fetch_topic_descriptions(
            self,
    ) -> Annotated[str, "A stringified dictionary of topics and their descriptions"]:
        topics_dict = vector_store_manager.get_descriptions()
        return str(topics_dict)

    @kernel_function(name="determine_relevant_topics",
                     description="Find the most relevant topics for the query")
    async def determine_relevant_topics(
            self,
            kernel: Kernel,
            query: Annotated[str, "The user query"],
            topics_descriptions: Annotated[str, "A stringified dictionary of topics and descriptions"]
    ) -> Annotated[str, "Stringified list of most relevant topic names, or 'general' if none are applicable"]:

        topics_dict = ast.literal_eval(topics_descriptions)
        prompt = f"""
        Given the following topics and their descriptions: {topics_dict},
        compare the content of the query with the descriptions
        of the topics and select the ones most relevant to
        answering the query:"{query}".
        Be sure to ONLY return a Python list string format like: ['Topic1', 'Topic2'].
        If none are applicable, return ['general'].
        """

        settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
        response = await kernel.invoke_prompt(
            function_name="determine_relevant_topics",
            plugin_name="QueryResponse",
            prompt=prompt,
            settings=settings,
        )
        # print("Relevant topics discussion: ",response)
        return response

    @kernel_function(name="retrieve_context_chunks",
                     description="Retrieve relevant chunks from Pinecone indices, including images.")
    async def retrieve_context_chunks(
            self,
            kernel: Kernel,
            topics: Annotated[str, "Stringified list of relevant topic names"],
            query: Annotated[str, "The user query"]
    ) -> Annotated[str, "Stringified list of retrieved text/image descriptions and image paths"]:
        """
        Retrieves the top-k most relevant chunks from the relevant topics' Pinecone indices.
        Identifies whether each retrieved vector is a text-chunk or an image-description.
        If an image, stores its file path for retrieval.
        """
        general_knowledge_request = "No relevant context found"

        # 🟢 Ensure topics list format is correct
        list_pattern = r"\[\s*(?:'[^']*'(?:\s*,\s*'[^']*')*)?\s*\]"
        def find_valid_list(text):
            match = re.search(list_pattern, text)
            return match.group(0) if match else None

        found_list = find_valid_list(topics)
        if topics == "['general']" or not found_list:
            return general_knowledge_request  # Fall back to general LLM knowledge

        found_list = ast.literal_eval(found_list)
        context_texts = []
        image_paths = []

        existing_indexes = vector_store_manager.list_indexes()  # Ensure only existing indexes are queried
        for topic in found_list:
            if topic not in existing_indexes:
                continue  # Skip missing index
            metadata_list = vector_store_manager.query_at_index(topic, query)
            for metadata in metadata_list:
                chunk_type = metadata.get("type", "text")  # Default to text
                content = metadata.get("content", "")
                # source_doc = metadata.get("source", "")
                image_path = metadata.get("file_path")
                # print(f"Source Document: {source_doc}")
                if chunk_type == "image":
                    # 🟢 Image descriptions get tagged for retrieval
                    image_paths.append(image_path)  # Save path for image retrieval
                context_texts.append(f"{content}\n File source: {image_path}")
        # Return both text and images
        response = str({"text_chunks": context_texts, "image_paths": image_paths}) if found_list else general_knowledge_request
        #print("Chunks: ", response)
        return response

    @kernel_function(name="answer_query",
                     description="Answer the user query with retrieved context, including images if available.")
    async def answer_query(
            self,
            kernel: Kernel,
            query: Annotated[str, "The user query"],
            retrieved_data: Annotated[str, "Stringified dictionary containing relevant text_chunks and image_paths"]
    ) -> Annotated[str, "Final answer to the user query"]:
        # Initialize general knowledge prompts in case no
        # relevant topics are available to answer the query
        general_knowledge_request = "No relevant context found"
        prompt = f"Answer the following question using your general knowledge:\n\nQuery: {query}"
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
        if retrieved_data == general_knowledge_request:
            response = await kernel.invoke_prompt(
                function_name="answer_query",
                plugin_name="QueryResponse",
                prompt=prompt,
                settings=settings,
            )
            return response
        else:
            # Parse the dictionary from the previous method
            retrieved_dict = ast.literal_eval(retrieved_data)
            text_chunks = retrieved_dict.get("text_chunks", [])
            image_paths = retrieved_dict.get("image_paths", [])

            # Locate and encode images to feed into GPT-4-Turbo
            encoded_images = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        encoded_img = encode_image(img_path)#base64.b64encode(img_file.read()).decode("utf-8")
                        encoded_images.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                        )
            # Build the prompt to answer the query & use
            # relevant information
            formatted_context = "\n\n".join(text_chunks)
            text_only_prompt = f"""
            Use the following retrieved information as context to intelligently
            answer the user's query. If the user asks for a source, please list
            the file path or image name associated with it.
            
            Retrieved information:
            {formatted_context}
            
            Full conversation & User Query: {query}
            """
            # If there are encoded images, use the OpenAI api.
            # Otherwise, use Semantic Kernel
            if encoded_images:
                prompt = text_only_prompt + "\n\n Please also use the retrieved visual images to aid in answering the query."
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + encoded_images
                    }
                ]
                response = turbo_client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages
                )
                response = response.choices[0].message.content
            else:
                prompt = text_only_prompt
                response = await kernel.invoke_prompt(
                    function_name="answer_query",
                    plugin_name="QueryResponse",
                    prompt=prompt,
                    settings=settings,
                )
            return response

    @kernel_function(name="format_response",
                     description="Takes the chatbot response and formats its text to be proper markdown format, which includes converting any latex/mathematical expressions into proper markdown math expressions.")
    async def format_response(
            self,
            kernel: Kernel,
            raw_response: Annotated[str, "The current, unformatted string response to a user's query"],
    ) -> Annotated[str, "The final response to the user query after making it markdown format"]:
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id="chat")
        prompt = f"""
        Please format the below Raw Chatbot response to a user's query
        so that it follows proper markdown format. If there appear to be
        mathematical expressions present, be sure to use $...$ for boxing
        in inline math expressions or $$...$$ for block math displays.
        Additionally, be sure to mark any headers as appropriate with #'s
        to indicate how big they are. If no mathematical statements or
        text or bullet headers are detected, don't provide special
        formatting. 
        
        Raw, unformatted response:
        {raw_response}
        """
        response = await kernel.invoke_prompt(
            function_name="format_response",
            plugin_name="QueryResponse",
            prompt=prompt,
            settings=settings,
        )
        return response

# === Add Plugins to the Kernel ===
kernel.add_plugin(QueryPlugin(), plugin_name="QueryResponse",
                  description="""
                  For question-answering related functions 
                  for identifying and selecting relevant 
                  topics for a topic-selection, retrieval of 
                  relevant content for context based on
                  those selected , answering 
                  user queries, and formatting the responses"""
                  )
kernel.add_plugin(TextPlugin(), plugin_name="text")


# === Query-Answering Pipeline Methods ===
async def run_query_pipeline(user_query: str):
    """Runs the query pipeline while keeping track of previous messages using SK memory."""
    # 🟢 Convert chat history to formatted prompt
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
    # 🟢 Create and run planner with memory-aware prompt
    goal_prompt = f"""
    Ingest the prior conversation and the current user query,
    then find the list of topics it could be related to,
    identify the most relevant topics to the user's query
    (including text and potentially images), and finally answer
    the said query using the retrieved information as context.
    Then make sure the final response is in proper markdown
    format and cleaned up for display.
    """
    plan = await planner.create_plan(goal_prompt)
    execution_result = await plan.invoke(kernel, {"query": full_prompt})
    # 🟢 Store the user's query and assistant's response in SK's memory
    chat_history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=execution_result.value))
    # print(execution_result.value)
    return execution_result.value

def run_query(user_query: str):
    """Wrapper method for inputting a user's question to the chatbot"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(run_query_pipeline(user_query))
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
