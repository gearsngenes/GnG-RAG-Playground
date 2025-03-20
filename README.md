# Welcome to Gears N' Genes RAG Playground
## Synopsis and Motivation
This repository provides a flask-api based application that provides a basic chatbot interface as well as an interface to configure your own Subject Matter Experts (SME's) for any topic or documentation you provide. As of right now, you can upload various types of images and documents, which you can then allow the chatbot to perform RAG (retrieval augmented generation) on to provide you contextually relevant answers.

For instance, if you are, say, a teacher trying to provide additional assistance for your students to understand the course content outside of class hours, you can configure this playground to ingest class presentations or textbooks you use, and it can use those documents to provide targeted answers to questions students might ask.

## The Tech Stack
As mentioned earlier, the API runs on Flask for the backend of this interface, but other noteworthy platforms/api's to take note of include:

- **Semantic Kernel**: Semantic Kernel is an intelligent planner library that can take in natural language descriptions of plans we want to execute, and intelligently creates an order of in which to execute specific methods, without us having to explicitly define this order of method calls. This is also the backbone of how the chatbot knows what topics and types of documents to use for reference before actually retrieving them. 
- **Pinecone**: This is the vector store database that we use to store our documents that we want to use for our RAG model. For each general "topic" we define, we create a specific corresponding Pinecone index. When we then upload files for our RAG model to use for reference, we first break the document text into smaller pieces or "chunks" that we then retrieve for additional context to answer queries later on.
- **OpenAI**: Ultimately Semantic Kernel is built on the OpenAI API, and I also make occassional calls to the OpenAI API directly instead of semantic kernel, particularly when generating descriptions for uploaded images for the RAG to use.