import ast
import os
from urllib.parse import quote

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from pinecone_utils import vector_store_manager
from helpers import encode_image, UPLOAD_FOLDER

# === Global message history ===
_message_history = []

def clear_chat_memory():
    _message_history.clear()

def get_chat_history():
    return _message_history.copy()

def format_history():
    """Serializes chat history into a readable format for prompts."""
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in _message_history)


# === LLM Setup ===
llm = ChatOpenAI(model="gpt-4o")


# === Topic Selector Prompt ===
topic_selector_prompt = PromptTemplate.from_template("""
You are reviewing a conversation to determine what topics are most relevant to answering the user's latest query.

Conversation so far:
{chat_history}

Available topics and descriptions:
{descriptions}

Return a Python list formatted like: ['topic1', 'topic2']. Return ['general'] if none are relevant.
""")

topic_selector_chain = topic_selector_prompt | llm


# === Answer Prompt ===
answer_prompt = PromptTemplate.from_template("""
You are a Markdown-ready assistant responding to a conversation.

Use ONLY the following text chunks and image descriptions to answer the latest user question in the context of the full discussion.

Each chunk and description has a source markdown link to refer to the original source from when generating a response. Follow these citation rules:
- If you use information from a specific source from the context chunks to generate a response, then you MUST a) explicitly refer to the name of the source file you are referring, AND b) incorporate the corresponding markdown link into the response to provide the user a way to view the source.
- You can only use a specific markdown link once, the first time you explicitly mention the source file that the content came from. Do NOT repeatedly display or link to the same file every time. 
- If your source file that your information comes from is an image description, you MUST explicitly use a markdown link, and render the image by placing a `!` in front of the link to have the image fully render in the response.
- If the user's query closely matches the description of an image, or implies or directly requests some visual, prioritize matching images to their request and fully rendering them.
- Only use information contained within the text chunks and image descriptions — do not rely on general knowledge unless it is to fill in the gaps of what would be obvious knowledge or to mbest match information to the user's query that doesn't exactly match the wordings available.

---
Chunks:
{context}

---
Conversation History:
{chat_history}

---
Respond to the latest user query/statement.
""")

answer_chain = answer_prompt | llm


# === Chunk Retrieval Logic ===
def retrieve_chunks(topics, query, use_general_knowledge=True):
    if topics == ['general']:
        return "No relevant context found" if use_general_knowledge else "no_information_found"

    context_texts = []
    image_paths = []

    for topic in topics:
        if topic not in vector_store_manager.list_indexes():
            continue

        # Get top 3 TEXT chunks
        text_results = vector_store_manager.query_at_index(
            index_name=topic,
            query=query,
            top_k=5,
            filter={"type": {"$eq": "text"}}
        )

        for metadata in text_results:
            content = metadata.get("content", "")
            file_path = metadata.get("file_path").replace("\\", "/")
            filename = os.path.basename(file_path)

            relative_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
            url_path = f"/{UPLOAD_FOLDER}/{quote(relative_path)}"
            markdown_link = f"[{filename}]({url_path})"
            context_texts.append(f"{content}\nSource URL: {markdown_link}")

        # Get top 3 IMAGE descriptions
        image_results = vector_store_manager.query_at_index(
            index_name=topic,
            query=query,
            top_k=5,
            filter={"type": {"$eq": "image"}}
        )

        for metadata in image_results:
            content = metadata.get("content", "").replace("\n","")
            file_path = metadata.get("file_path").replace("\\", "/")

            relative_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
            url_path = f"/{UPLOAD_FOLDER}/{quote(relative_path)}"
            markdown_link = f"![{content}]({url_path})"
            image_paths.append(markdown_link)

    if not context_texts and not image_paths:
        return "No relevant context found" if use_general_knowledge else "no_information_found"
    final_context = "<TEXT CHUNKS>\n" + "\n\n".join(context_texts) + "\n\n<IMAGE DESCRIPTIONS>\n" + "\n\n".join(image_paths)
    print("FINAL context retrieved: ", final_context)
    return final_context


# === Retrieval → Answer chain
def retrieve_context(inputs):
    context = retrieve_chunks(
        topics=inputs["topics"],
        query=inputs["query"],
        use_general_knowledge=inputs["use_general_knowledge"]
    )
    return {
        "context": context,
        "query": inputs["query"],
        "chat_history": format_history()
    }

retrieve_chain = RunnableLambda(retrieve_context)
rag_chain = retrieve_chain | answer_chain


# === Main Query Handler ===
def run_langchain_query(query, topics=None, use_general_knowledge=True):
    _message_history.append({"role": "user", "content": query})

    if not topics:
        if use_general_knowledge:
            topics = ['general']
        else:
            descriptions = vector_store_manager.get_descriptions()
            topic_output = topic_selector_chain.invoke({
                "query": query,
                "descriptions": descriptions,
                "chat_history": format_history()
            })
            try:
                topics = ast.literal_eval(topic_output.content)
            except Exception:
                topics = ['general']

    if topics == ['general']:
        if not use_general_knowledge:
            assistant_msg = (
                "❌ Sorry, we couldn’t find any relevant topics or matching content "
                "in your uploaded documents to answer your question."
            )
        else:
            assistant_msg = llm.invoke(f"Answer this using general knowledge:\n\n{format_history()}\n\nNew query:\n{query}").content
    else:
        result = rag_chain.invoke({
            "query": query,
            "topics": topics,
            "use_general_knowledge": use_general_knowledge
        })
        assistant_msg = result.content if hasattr(result, "content") else result

    _message_history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

"""
Uncomment the below lines if you want to test the pipeline locally
"""
# if __name__ == "__main__":
#     query = input("Enter your query:\n> ")
#     print(run_langchain_query(query))
