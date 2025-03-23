from flask import Flask, request, jsonify, render_template
from pinecone_utils import vector_store_manager
from rag_kernel import run_query, clear_sk_memory, get_chat_history
import os
import shutil
from helpers import (UPLOAD_FOLDER,
                     DOC_EXTENSIONS,
                     IMG_EXTENSIONS,
                     extract_text,
                     generate_gpt4_description,
                     extract_images_from_pdf,
                     extract_images_from_docx,
                     extract_images_from_pptx)

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
    """Lists the names of uploaded files for a given topic/index with their embedding status."""
    data = request.get_json()
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    folder_path = os.path.join(UPLOAD_FOLDER, index_name)
    if not os.path.exists(folder_path):
        return jsonify({"files": []})

    files = os.listdir(folder_path)
    file_info = []

    for file_name in files:
        embedded = vector_store_manager.is_embedded(index_name, file_name)
        file_info.append({"name": file_name, "embedded": embedded})

    return jsonify({"files": file_info})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    index_name = request.form.get('index_name')
    file = request.files.get('file')
    if not index_name:
        return jsonify({"error": "Invalid index selection."}), 400
    if not file:
        return jsonify({"error": "No file provided."}), 400

    topic_dir = os.path.join(UPLOAD_FOLDER, index_name)
    os.makedirs(topic_dir, exist_ok=True)

    #file_base_name = os.path.splitext(file.filename)[0]
    document_dir = os.path.join(topic_dir, file.filename)
    os.makedirs(document_dir, exist_ok=True)
    file_path = os.path.join(document_dir, file.filename)
    file.save(file_path)
    images_saved = []
    if ("." + file.filename.split(".")[-1]) not in IMG_EXTENSIONS:
        document_image_dir = os.path.join(document_dir, "images")
        os.makedirs(document_image_dir, exist_ok=True)
        if "txt" != file.filename.split(".")[-1]:
            ext = file_path.lower()
            if ext.endswith(".pdf"):
                images_saved = extract_images_from_pdf(file_path, document_image_dir)
            elif ext.endswith(".docx"):
                images_saved = extract_images_from_docx(file_path, document_image_dir)
            elif ext.endswith(".pptx"):
                images_saved = extract_images_from_pptx(file_path, document_image_dir)
    else:
        images_saved = [file_path]
    return jsonify({
        "message": f"Document '{file.filename}' and {len(images_saved)} images saved successfully."
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
    print("files_to_embed: ", files_to_embed)
    for file_name in files_to_embed:
        file_dir = os.path.join(topic_path, file_name)
        file_path = os.path.join(file_dir, file_name)
        ext = "."+ file_name.split(".")[-1]
        print(f"File: {file_name}, Dir: {file_dir}, Path: {file_path}, Extension: {ext}")
        # if not os.path.isfile(file_dir):
        #     continue
        image_descriptions = []
        image_paths = []
        if ext in DOC_EXTENSIONS:
            try:
                text_chunks = extract_text(file_path, chunk_size)
                total_text_vectors += len(text_chunks)
                file_paths = [file_path] * len(text_chunks)
                if text_chunks:
                    vector_store_manager.upsert_vectors(index_name, file_name, file_paths, text_chunks, "text")
            except Exception as e:
                print(f"Error extracting text from {file_name}: {e}")
                continue
            images_dir = os.path.join(file_dir, "images")
            if os.path.exists(images_dir):
                for image_file in os.listdir(images_dir):
                    image_path = os.path.join(images_dir, image_file)
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        description = generate_gpt4_description(image_path)
                        image_descriptions.append(description)
                        image_paths.append(image_path)
        elif ext in IMG_EXTENSIONS:
            print("Image path: ", file_path)
            image_paths = [file_path]
            image_descriptions = [generate_gpt4_description(file_path)]
            print("Generated description: ", image_descriptions)
        if image_descriptions:
            vector_store_manager.upsert_vectors(index_name, file_name, image_paths, image_descriptions, "image")
            total_image_vectors += len(image_descriptions)

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

    for file_name in files_to_unembed:
        try:
            vector_store_manager.delete_vectors_by_source(index_name, file_name)
        except Exception as e:
            return jsonify({"error": f"Failed to unembed '{file_name}': {str(e)}"}), 500

    return jsonify({"message": f"Vectors for selected files have been removed from '{index_name}'."})

@app.route('/delete_files', methods=['POST'])
def delete_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_delete = data.get("files", [])
    if not index_name or not files_to_delete:
        return jsonify({"error": "Index name and files to delete are required."}), 400

    for file_name in files_to_delete:
        document_dir = os.path.join(UPLOAD_FOLDER, index_name, file_name)
        if os.path.exists(document_dir):
            shutil.rmtree(document_dir)
        vector_store_manager.delete_vectors_by_source(index_name, file_name)

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
    if not query_text:
        return jsonify({"error": "Query text is required."}), 400
    response = run_query(query_text)
    return jsonify({"response": str(response)})

@app.route('/load_conversation', methods=['GET'])
def get_chat_history_api():
    history = get_chat_history()
    return jsonify({"chat_history": history})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    clear_sk_memory()
    return jsonify({"message": "Chat history cleared."})


# === To start the application ===
if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)
