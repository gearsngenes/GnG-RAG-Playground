from flask import Flask, request, jsonify, render_template, send_from_directory
#from pinecone_utils import vector_store_manager
from qdrant_utils import vector_store_manager
#from rag_kernel import run_query, clear_sk_memory, get_sk_chat_history
#from rag_chain import run_langchain_query, get_langchain_chat_history, clear_langchain_chat_memory
from rag_slm import run_slm_query, get_chat_history, clear_chat_memory
import os
import shutil
import json
from helpers import (UPLOAD_FOLDER,
                     DOC_EXTENSIONS,
                     IMG_EXTENSIONS,
                     extract_text,
                     extract_images_from_pdf,
                     extract_images_from_docx,
                     extract_images_from_pptx)
import urllib.parse

app = Flask(__name__)
app.secret_key = 'supersecretkey'

#===Page-Wide Rendering===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/manage_topics')
def manage_topics():
    return render_template('manage_topics.html')

#===Topic-Management====
"""
The following methods are used for tracking, creating, &
removing pinecone indexes on Pinecone. These methods include

-   list_indexes()
        List existing indexes or "topics" for
        dropdown lists on the frontend.
        
-   get_index_description() & update_index_description()
        retrieve the general description for a
        given index, allowing for it to be
        edited and saved if changes are needed.
        
-   create_index()
        For creating new pinecone indices for a topic.
        Its description is saved in a Table of Contents
        index, separate from the other indexes, while a
        whole index with that name is created separately.
        
-   delete_index()
        For deleting an index on pinecone and removing
        its description from the Table of Contents
        index, ensuring total consistency
"""
@app.route('/list_indexes', methods=['GET'])
def list_indexes():
    indexes = vector_store_manager.list_indexes()
    return jsonify(indexes)

@app.route('/get_index_description', methods=['POST'])
def get_index_description():
    data = request.json
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        description = vector_store_manager.get_index_description(index_name)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_index_description', methods=['POST'])
def update_index_description():
    data = request.json
    index_name = data.get("index_name")
    new_description = data.get("description", "")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        vector_store_manager.upsert_metadata(index_name, new_description)
        return jsonify({"message": f"Description for '{index_name}' updated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create_index', methods=['POST'])
def create_index():
    data = request.json
    index_name = data.get("index_name")
    description = data.get("description", "")
    if not index_name or not index_name.islower() or not all(c.isalnum() or c == '-' for c in index_name):
        return jsonify({"error": "Index name must be lowercase, alphanumeric, or contain '-' only."}), 400
    try:
        vector_store_manager.create_index(index_name)
        vector_store_manager.upsert_metadata(index_name, description)
        return jsonify({"message": f"Index '{index_name}' created successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_index', methods=['POST'])
def delete_index():
    data = request.json
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        vector_store_manager.delete_index(index_name)
        return jsonify({"message": f"Index '{index_name}' deleted successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#===Document Management===
"""
The following methods are used for managing individual
& multiple files that have been uploaded locally for
a particular topic:

-   list_uploaded_files()
        This lists all files that have been uploaded
        for a specific topic.
        
-   upload_document()
        This method takes in a file from the frontend
        and saves it locally within a directory with
        the file's name as a subdirectory of the
        topic it was submitted under. Additionally,
        pdf's, docx's, and pptx's get an images/
        subfolder where any images in a document
        are extracted
        
-   embed_files()
        This method extracts the text of selected
        documents, and breaks them into chunks. Each
        chunk is then embedded as a vector onto
        the corresponding pinecone index. For images,
        we first create a description using the
        GPT-4-Turbo model, we can then embedd like
        other text chunks. Appropriate metadata
        is stored so that we can retrieve the text,
        source file, or image, as well.

-   delete_files()
        This method deletes vectors from selected
        files on the local directory, as well as
        any vectors they were embedded in on
        Pinecone. This ensures that we cannot
        use them for context, going forward.
"""
@app.route('/list_uploaded_files', methods=['POST'])
def list_uploaded_files():
    """Lists the names of uploaded files (cleaned) and their embedding status."""
    data = request.get_json()
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400

    topic_path = os.path.join(UPLOAD_FOLDER, index_name)
    if not os.path.exists(topic_path):
        return jsonify({"files": []})

    file_info = []

    for dirname in os.listdir(topic_path):
        document_dir = os.path.join(topic_path, dirname)
        if not os.path.isdir(document_dir):
            continue

        # Derive filename from directory name
        clean_filename = dirname.replace("_dot_", ".")
        file_path = os.path.join(document_dir, clean_filename)

        if os.path.exists(file_path):
            embedded = vector_store_manager.is_embedded(index_name, clean_filename)
            file_info.append({"name": clean_filename, "embedded": embedded})

    return jsonify({"files": file_info})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    index_name = request.form.get('index_name')
    file = request.files.get('file')
    user_description = request.form.get('image_description', '').strip()

    if not index_name:
        return jsonify({"error": "Invalid index selection."}), 400
    if not file:
        return jsonify({"error": "No file provided."}), 400

    # === 🆕 Sanitize filename and directory name
    original_filename = file.filename
    clean_filename = original_filename.replace(" ", "_")
    dir_name = clean_filename.replace(".", "_dot_")

    topic_dir = os.path.join(UPLOAD_FOLDER, index_name)
    document_dir = os.path.join(topic_dir, dir_name)
    os.makedirs(document_dir, exist_ok=True)

    file_path = os.path.join(document_dir, clean_filename)
    file.save(file_path)

    ext = os.path.splitext(clean_filename)[-1].lower()
    images_saved = []

    if ext not in IMG_EXTENSIONS:
        # === 📄 Document — extract embedded images if applicable
        document_image_dir = os.path.join(document_dir, "images")
        os.makedirs(document_image_dir, exist_ok=True)

        if ext.endswith(".pdf"):
            images_saved = extract_images_from_pdf(document_dir, file_path, document_image_dir)
        elif ext.endswith(".docx"):
            images_saved = extract_images_from_docx(document_dir, file_path, document_image_dir)
        elif ext.endswith(".pptx"):
            images_saved = extract_images_from_pptx(document_dir, file_path, document_image_dir)
    else:
        # === 🖼 Standalone image
        images_saved = [file_path]
        if user_description:
            alt_map_path = os.path.join(document_dir, "alt_image_map.json")
            with open(alt_map_path, "w", encoding="utf-8") as f:
                json.dump([{
                    "path": file_path,
                    "alt_text": user_description
                }], f, indent=4)

    return jsonify({
        "message": f"Document '{clean_filename}' and {len(images_saved)} images saved successfully."
    })

@app.route('/embed_files', methods=['POST'])
def embed_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_embed = list(data.get("files", []))
    chunk_size = int(data.get("chunk_size", 500))

    if not index_name or not files_to_embed:
        return jsonify({"error": "Index name and files are required."}), 400

    topic_path = os.path.join(UPLOAD_FOLDER, index_name)
    total_text_vectors = 0
    total_image_vectors = 0

    for clean_filename in files_to_embed:
        dir_name = clean_filename.replace(".", "_dot_")
        file_dir = os.path.join(topic_path, dir_name)
        file_path = os.path.join(file_dir, clean_filename)
        ext = os.path.splitext(clean_filename)[-1].lower()

        # === TEXT ===
        if ext in DOC_EXTENSIONS:
            try:
                text_chunks = extract_text(file_path, chunk_size)
                total_text_vectors += len(text_chunks)
                file_paths = [file_path] * len(text_chunks)

                if text_chunks:
                    vector_store_manager.upsert_vectors(
                        index_name=index_name,
                        src_doc=clean_filename,
                        file_paths=file_paths,
                        chunks=text_chunks,
                        embed_type="text"
                    )
            except Exception as e:
                print(f"Error extracting text from {clean_filename}: {e}")

        # === IMAGE ===
        alt_map_path = os.path.join(file_dir, "alt_image_map.json")
        if os.path.exists(alt_map_path):
            try:
                with open(alt_map_path, "r", encoding="utf-8") as f:
                    alt_images_info = json.load(f)

                image_paths = []
                image_descriptions = []

                for entry in alt_images_info:
                    img_path = entry.get("path")
                    alt_text = entry.get("alt_text")
                    if os.path.exists(img_path) and alt_text:
                        image_paths.append(img_path)
                        image_descriptions.append(alt_text)

                if image_descriptions:
                    vector_store_manager.upsert_vectors(
                        index_name=index_name,
                        src_doc=clean_filename,
                        file_paths=image_paths,
                        chunks=image_descriptions,
                        embed_type="image"
                    )
                    total_image_vectors += len(image_descriptions)

            except Exception as e:
                print(f"Error reading alt_image_map.json for {clean_filename}: {e}")

    return jsonify({
        "message": f"Embedding complete: {total_text_vectors} text chunks and {total_image_vectors} images processed."
    })

@app.route('/unembed_files', methods=['POST'])
def unembed_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_unembed = list(data.get("files", []))

    if not index_name or not files_to_unembed:
        return jsonify({"error": "Index name and files to unembed are required."}), 400

    for clean_filename in files_to_unembed:
        try:
            vector_store_manager.delete_vectors_by_source(index_name, clean_filename)
        except Exception as e:
            return jsonify({"error": f"Failed to unembed '{clean_filename}': {str(e)}"}), 500

    return jsonify({"message": f"Vectors for selected files have been removed from '{index_name}'."})

@app.route('/delete_files', methods=['POST'])
def delete_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_delete = data.get("files", [])
    if not index_name or not files_to_delete:
        return jsonify({"error": "Index name and files to delete are required."}), 400

    for clean_filename in files_to_delete:
        dir_name = clean_filename.replace(".", "_dot_")
        document_dir = os.path.join(UPLOAD_FOLDER, index_name, dir_name)

        if os.path.exists(document_dir):
            shutil.rmtree(document_dir)

        vector_store_manager.delete_vectors_by_source(index_name, clean_filename)

    return jsonify({"message": f"Selected files and associated data have been deleted from '{index_name}'."})

#===Conversation Methods===
"""
For getting questions from the frontend, generating
responses to the questions using semantic kernel,
and clearing the chat history when we are finished.
"""
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get("query")
    topics = list(data.get("topics", []))

    if not query_text:
        return jsonify({"error": "Query text is required."}), 400

    #result = run_query(query_text, topics)#
    result = run_slm_query(query_text, topics)
    #print(result['response'])
    return jsonify(result)

@app.route(f'/{UPLOAD_FOLDER}/<path:filename>')
def serve_uploaded_file(filename):
    safe_path = urllib.parse.unquote(filename)
    return send_from_directory(UPLOAD_FOLDER, safe_path)

@app.route('/load_conversation', methods=['GET'])
def get_chat_history_api():
    #history = get_sk_chat_history()
    history = get_chat_history()
    return jsonify({"chat_history": history})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    #clear_sk_memory()
    clear_chat_memory()
    return jsonify({"message": "Chat history cleared."})

# === To start the application ===
if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)
