import os
from urllib.parse import quote
from llama_cpp import Llama
from qdrant_utils import vector_store_manager
from helpers import UPLOAD_FOLDER
from prompts import full_prompt_phi4, full_prompt_stablelm, general_knowledge_prompt

# === Model Setup ===
model_configs = {
    "phi4":
        {
            "repo_id":"unsloth/phi-4-GGUF",
            "filename":"phi-4-Q4_K_M.gguf",
            "n_ctx":8000,
            "max_ctxt":128000,
            "top_k": 5,
            "prompt":full_prompt_phi4
        },
    "Mistral-7B-DPO":
        {
            "repo_id":"NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
            "filename":"Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf",
            "n_ctx":8000,
            "max_ctxt":32000,
            "top_k": 5,
            "prompt":full_prompt_phi4
        },
    "Stable LM 2":
        {
            "repo_id":"stabilityai/stablelm-2-zephyr-1_6b",
            "filename":"stablelm-2-zephyr-1_6b-Q5_K_M.gguf",
            "n_ctx":2500,
            "max_ctxt":4096,
            "top_k": 2,
            "prompt":full_prompt_stablelm
        }
}
MODEL_NAME = "Mistral-7B-DPO"
MODEL_CONFIG = model_configs.get(MODEL_NAME)

REPO_ID = MODEL_CONFIG.get("repo_id")
MODEL_FILENAME = MODEL_CONFIG.get("filename")
N_CTX = MODEL_CONFIG.get("n_ctx")
TOP_K = MODEL_CONFIG.get("top_k")

RAG_PROMPT_TEMPLATE = MODEL_CONFIG.get("prompt")
MAX_TOKENS = 1000

'''
Phi-4
Context Window: 128K
Best performing large-context model that I've tried out, but it is 
somewhat demanding with regards to cpu and the time it takes. For
small projects, I'd recommend keeping it to around 16K token for 
context for it to use, unless you have GPU or more RAM to run it
---

stabilityai/stablelm-2-zephyr-1_6b
INCREDIBLY FAST, but has a limited 4K token window and isn't as accurate.
Great for building simple chatbots and using its general knowledge, but it
is not well suited for RAG models that require large context windows to
answer questions about documents. Best to use for simple q&a tasks that
don't require a lot of reasoning.
It is relatively weak in terms of following prompt instructions, as it
doesn't properly cite the sources that it uses when I asked, and it
still occasionally uses its own general knowledge not specified in the context.

Maximum Token Window: 4K, but better performing to use around 2K

The prompt template method associated with it does NOT use conversation context
---

Nous-Hermes-2 Mistral-7B-DPO
Context Window: 32K
It is the second best performing model of the three listed here,
comparatively similar in performance to Phi-4, albeit it loses
slightly in accuracy per some of the questions I've tested it
on with regards to my RAG content. Perfectly suitable for any
standard chatbot, however, as it is able to generate responses
almost twice as fast as Phi-4. More testing to be done to
confirm its effectiveness.
'''

llm = Llama.from_pretrained(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME,
    n_ctx=N_CTX
)
# === Chat History Management ===
_message_history = []

def clear_chat_memory():
    """Clears the internal message history buffer."""
    _message_history.clear()

def get_chat_history():
    """Returns current chat history as a list of role/content dictionaries."""
    return [{"role": msg["role"], "content": msg["content"]} for msg in _message_history]

def format_history():
    """Returns a formatted string of the chat history for prompt injection."""
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in _message_history)

def retrieve_chunks(topics, query):
    """
    Retrieve relevant text and image chunks for selected topics.
    Returns a formatted string of context, or "general_knowledge_only" if topic is "general".
    """
    if topics == ['general']:
        return "general_knowledge_only"

    context_texts = []
    image_paths = []

    for topic in topics:
        if topic not in vector_store_manager.list_indexes():
            continue

        # === TEXT RESULTS ===
        text_results = vector_store_manager.query_at_index(
            index_name=topic,
            query=query,
            top_k=TOP_K,
            filter={"type": {"$eq": "text"}}
        )

        for metadata in text_results:
            content = metadata.get("content", "")
            file_path = metadata.get("file_path", "").replace("\\", "/")
            filename = os.path.basename(file_path)
            rel_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
            url = f"/{UPLOAD_FOLDER}/{quote(rel_path)}"
            context_texts.append(f"{content}\nSource URL: [{filename}]({url})")

        # === IMAGE RESULTS ===
        image_results = vector_store_manager.query_at_index(
            index_name=topic,
            query=query,
            top_k=TOP_K,
            filter={"type": {"$eq": "image"}}
        )

        for i,metadata in enumerate(image_results):
            content = metadata.get("content", "").replace("\n", "")
            file_path = metadata.get("file_path", "").replace("\\", "/")
            filename = os.path.basename(file_path)
            rel_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
            url = f"/{UPLOAD_FOLDER}/{quote(rel_path)}"
            markdown_url = f"![{filename}]({url})"
            image_paths.append(f"Image {i}: {filename}\nURL:{markdown_url}\nDescription: {content}")

    if not context_texts and not image_paths:
        return "general_knowledge_only"

    result = "<TEXT CHUNKS>\n" + "\n\n".join(context_texts)
    if image_paths:
        result += "\n\n<IMAGE DESCRIPTIONS>\n" + "\n\n".join(image_paths)
    #print("RETRIEVED RESULTS:\n", result)
    return result

def run_slm_query(query, topics):
    # Format it into a string
    history = format_history()
    # Update history
    _message_history.append({"role": "user", "content": query})
    # Retrieve context if specific topics were selected
    context = "" if "general" in topics else retrieve_chunks(topics, query)
    # Construct prompt based on whether specific topics were selected
    full_prompt = general_knowledge_prompt(history, query) if "general" in topics else RAG_PROMPT_TEMPLATE(context, history, query)
    # Generate the result
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )
    # Parse final output
    response = result["choices"][0]["message"]["content"].strip()
    # Add response to history and return response
    _message_history.append({"role": "assistant", "content": response})
    return {"response": response}

# === CLI Debug Driver ===
if __name__ == "__main__":
    print("🧪 SLM RAG Playground — CLI Test Mode")
    print("Available topics:", vector_store_manager.list_indexes()+["general"])
    print("Type your query and a comma-separated list of topics.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("📝 Query: ")
        if query.lower() == "exit":
            break

        topic_str = input("📚 Topics (comma-separated): ")
        topics = [t.strip() for t in topic_str.split(",") if t.strip()]

        result = run_slm_query(query, topics)
        print("\n🤖 Assistant (response):\n", result["response"])
