# Gears N' Genes RAG Playground (Local SLM + Qdrant Edition)

## 🧩 Overview

This is a self-contained Retrieval-Augmented Generation (RAG) playground powered by:

- **Local embedding models** (`intfloat/e5-small-v2`)
- **Qdrant** for local vector search
- **Small Language Models (SLMs)** like `phi-4` running via `llama-cpp`
- **Flask** for a lightweight API and interactive web UI

Users can upload documents (PDF, DOCX, PPTX, TXT) or images (JPG, PNG), organize them into topics, embed their contents for semantic search, and query them through a local model with markdown-rich responses and image rendering.

---

## ⚙️ Tech Stack

| Component       | Technology                  |
|----------------|-----------------------------|
| Backend        | Flask                       |
| Embeddings     | sentence-transformers       |
| Vector Store   | Qdrant (local instance)     |
| LLM Inference  | llama-cpp (GGUF models)     |
| Frontend       | HTML + Javascript + CSS     |
| Image Support  | Markdown + Alt-text mapping |

---

## 🗃 File Structure

```
├── app.py                  # Flask server & API endpoints
├── rag_slm.py              # SLM-based retrieval & markdown-based response logic
├── qdrant_utils.py         # Qdrant vector DB manager (embedding, querying, indexing)
├── helpers.py              # File/image processing, chunking, GPT alt-text generation
├── prompts.py              # Prompt templates for markdown-enforced responses
├── templates/
│   ├── index.html          # Chat UI
│   └── manage_topics.html  # Topic & file management UI
├── static/
│   ├── css/
│   │   ├── index.css
│   │   └── manage_topics.css
│   └── js/
│       ├── index.js
│       └── manage_topics.js
└── .env                    # Your environment config (see below)
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually ensure these are installed:

- flask
- qdrant-client
- sentence-transformers
- llama-cpp-python
- python-pptx, python-docx, PyPDF2

---

### 2. Setup Qdrant (Local Vector DB)

```bash
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

---

### 3. Prepare `.env` File

```env
UPLOAD_ROOT=uploads
OPENAI_API_KEY = 'your-api-key' # OPTIONAL, if you are working with images in PDF's
```

> ✅ No OpenAI or Pinecone API keys needed with the EXCEPTION of generating alt-text for images inside PDF's — this is a fully local deployment.

---

### 4. Launch the App

```bash
python app.py
```

Access the app at: [http://localhost:5000](http://localhost:5000)

---

## 🔍 Features

- ✅ Topic creation and metadata descriptions
- ✅ File uploads with automatic subdirectory naming
- ✅ Auto-extraction of embedded images from PDFs, DOCX, PPTX
- ✅ Alt-text generation for embedded images or user-input descriptions
- ✅ Embedding via `sentence-transformers` model
- ✅ Fast retrieval using Qdrant
- ✅ Query interface using `phi-4` via llama-cpp
- ✅ Structured markdown output with citation links & rendered images

---

## 🧠 Local Model Notes

By default, the app loads:

```python
llm = Llama.from_pretrained(
    repo_id="unsloth/phi-4-GGUF",
    filename="phi-4-Q4_K_M.gguf",
    n_ctx=3000
)
```

To use another GGUF-compatible model, replace the `repo_id` and `filename` in `rag_slm.py`.

---

## 🖼 Image Citation Format

Image files must include alt-text in one of two ways:

- Auto-generated via GPT-4-turbo for embedded images (if available)
- Manually entered for JPG/PNG uploads

These are then embedded alongside other text content and rendered with:

```markdown
![Image description](uploads/topic_dir/your_image.jpg)
```

---

## 🐳 Docker Instructions (Optional)

To deploy via Docker:

```bash
docker build -t gnrag .
docker run -d -p 5000:5000 --name rag-local gnrag
```

Then access the app at: `http://<your.local.ip>:5000`

---

## 🧪 Testing CLI (Optional)

To test queries in terminal with the local model:

```bash
python rag_slm.py
```

---

## 🧹 Future Ideas

- [ ] Multi-model support toggle (e.g. switch between phi-4, OpenChat, Mistral)
- [ ] RAG agent orchestration using LangGraph/AutoGen
- [ ] Enhanced PDF parsing with table/image recognition
- [ ] Authentication and multi-user isolation

---

## 📬 Feedback & Issues

Pull requests, suggestions, and bug reports are welcome! This project is designed for privacy-first local deployments, particularly for educators and researchers.
