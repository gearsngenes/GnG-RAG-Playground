import os
import re
from urllib.parse import quote
from llama_cpp import Llama
from pinecone_utils import vector_store_manager
from helpers import UPLOAD_FOLDER

# === Model Setup ===
N_CTX = 3000
MAX_TOKENS = 500
TOP_K = 5

llm = Llama.from_pretrained(
    repo_id="unsloth/phi-4-GGUF",
    filename="phi-4-Q4_K_M.gguf",
    # repo_id="unsloth/Phi-4-mini-reasoning-GGUF",
    # filename="Phi-4-mini-reasoning-Q4_K_M.gguf",
    n_ctx=N_CTX
)

# === Chat History ===
_message_history = []

def clear_chat_memory():
    _message_history.clear()

def get_chat_history():
    return [{"role": msg["role"], "content": msg["content"]} for msg in _message_history]

def format_history():
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in _message_history)

def retrieve_chunks(topics, query, use_general_knowledge=True):
    if topics == ['general']:
        return "No relevant context found" if use_general_knowledge else "no_information_found"

    context_texts = []
    image_paths = []

    for topic in topics:
        if topic not in vector_store_manager.list_indexes():
            continue

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

        image_results = vector_store_manager.query_at_index(
            index_name=topic,
            query=query,
            top_k=TOP_K,
            filter={"type": {"$eq": "image"}}
        )

        for metadata in image_results:
            content = metadata.get("content", "").replace("\n", "")
            file_path = metadata.get("file_path", "").replace("\\", "/")
            filename = os.path.basename(file_path)
            rel_path = file_path[len(f"{UPLOAD_FOLDER}/"):] if file_path.startswith(f"{UPLOAD_FOLDER}/") else file_path
            url = f"/{UPLOAD_FOLDER}/{quote(rel_path)}"
            markdown_url = f"![{filename}]({url})"
            image_paths.append(markdown_url + f"\nDescription: {content}")

    if not context_texts and not image_paths:
        return "No relevant context found" if use_general_knowledge else "no_information_found"

    result = "<TEXT CHUNKS>\n" + "\n\n".join(context_texts)
    if image_paths:
        result += "\n\n<IMAGE DESCRIPTIONS>\n" + "\n\n".join(image_paths)
    return result

def extract_think_response_sections(text):
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    response_match = re.search(r"<response>\s*(.*?)\s*</response>", text, re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""
    response = response_match.group(1).strip() if response_match else text.strip()
    return think, response

def run_slm_query(query, topics, use_general_knowledge=True):
    _message_history.append({"role": "user", "content": query})

    if not topics:
        response = "❌ Error: No topics provided. This version requires user-supplied topics."
        _message_history.append({"role": "assistant", "content": response})
        return {"think": "", "response": response}

    context = retrieve_chunks(topics, query, use_general_knowledge)

    if context == "no_information_found":
        if not use_general_knowledge:
            response = (
                "❌ Sorry, we couldn’t find any matching content from the selected topics "
                "to answer your question."
            )
            _message_history.append({"role": "assistant", "content": response})
            return {"think": "", "response": response}
        else:
            general_prompt = f"Answer the following using general knowledge:\n\n{format_history()}\n\nNew query:\n{query}"
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": general_prompt}],
                temperature=0.3,
                max_tokens=MAX_TOKENS
            )
            response = result["choices"][0]["message"]["content"].strip()
            _message_history.append({"role": "assistant", "content": response})
            return {"think": "", "response": response}
    from prompts import full_prompt_phi4
    full_prompt = full_prompt_phi4(context, format_history(), query)

    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1,
        max_tokens=MAX_TOKENS
    )

    full_output = result["choices"][0]["message"]["content"].strip()
    think, response = extract_think_response_sections(full_output)

    _message_history.append({"role": "assistant", "content": response})
    return {"think": think, "response": response}

# === CLI Test Driver ===
if __name__ == "__main__":
    print("🧪 SLM RAG Playground — CLI Test Mode")
    print("Available topics:", vector_store_manager.list_indexes())
    print("Type your query and a comma-separated list of topics.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("📝 Query: ")
        if query.lower() == "exit":
            break

        topic_str = input("📚 Topics (comma-separated): ")
        topics = [t.strip() for t in topic_str.split(",") if t.strip()]

        result = run_slm_query(query, topics)
        print("\n🤖 Assistant (thinking):\n", result["think"])
        print("\n🤖 Assistant (response):\n", result["response"])
