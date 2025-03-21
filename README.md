# Welcome to Gears N' Genes RAG Playground
## Synopsis and Motivation
This repository provides a flask-api based application that provides a basic chatbot interface as well as an interface to configure your own Subject Matter Experts (SME's) for any topic or documentation you provide. As of right now, you can upload various types of images and documents, which you can then allow the chatbot to perform RAG (retrieval augmented generation) on to provide you contextually relevant answers.

For instance, if you are, say, a teacher trying to provide additional assistance for your students to understand the course content outside of class hours, you can configure this playground to ingest class presentations or textbooks you use, and it can use those documents to provide targeted answers to questions students might ask.

## The Tech Stack
As mentioned earlier, the API runs on Flask for the backend of this interface, but other noteworthy platforms/api's to take note of include:

- **Semantic Kernel**: Semantic Kernel is an intelligent planner library that can take in natural language descriptions of plans we want to execute, and intelligently creates an order of in which to execute specific methods, without us having to explicitly define this order of method calls. This is also the backbone of how the chatbot knows what topics and types of documents to use for reference before actually retrieving them. 
- **Pinecone**: This is the vector store database that we use to store our documents that we want to use for our RAG model. For each general "topic" we define, we create a specific corresponding Pinecone index. When we then upload files for our RAG model to use for reference, we first break the document text into smaller pieces or "chunks" that we then retrieve for additional context to answer queries later on.
- **OpenAI**: Ultimately Semantic Kernel is built on the OpenAI API, and I also make occassional calls to the OpenAI API directly instead of semantic kernel, particularly when generating descriptions for uploaded images for the RAG to use.

## Files of Note
Here is the high-level breakdown of what files you need to know about in order to run this playground.

### app.py
This is the file that is running your Flask API. It is also this file that you will run in order to start the playground. Simply go to a python terminal and enter:
`python app.py`

### pinecone_utils.py
This is the file that supports the pinecone vector management. This is what handles the creation, deletion, and modification of indexes on PineCone and what stores the embedded versions of uploaded files. 

### rag_kernel.py
This is the file that handles Semantic Kernel logic with regards to actually retrieving chunks of contextually relevant information and answering user queries.

### helpers.py
This file contains various helper variables regarding OpenAI client objects as well as document processing methods such as text and image extraction.
