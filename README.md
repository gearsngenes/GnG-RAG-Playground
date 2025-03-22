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

## Setup Instructions
Before you can run the playground, you need to make sure you've set up the following:
1. Your OpenAI account, along with a corresponding API key
2. Your PineCone account, along with a corresponding API key

Next, after cloning this repository, you will need to make a `.env` file in the main directory where you can specify these two keys as well as the name of the uploads folder for where you want your index topics directories and their file directories to be.

Your `.env` file should be structured like so:
```
PINECONE_API_KEY = "your-pinecone-api-key"
OPENAI_API_KEY = "your-openai-api-key"
UPLOAD_ROOT = "uploads"
```
Be sure to replace the values with your actual api keys and your chosen folder name.

## GnG RAG Playground on Docker
If you'd like to have the app continuously running in the background, then there is a dockerfile that you can set up on your device's network. Make sure you have Docker installed on you device before running it.

First, go into the Playground's main directory (this repo's main directory) on a terminal, and enter the command:

`docker build -t <my-flask-app> .`

This will construct an image on Docker that will house the necessary components. It may take a while to initialize all the library requirements. Next, you need to run:

`docker run -d -p 5000:5000 --name <flask-container> <my-flask-app>`

Change the placeholder names to what you'd actually want them to be called.

This will create and run the container for which your application will be on. This will also make sure you don't need to have it running constantly on your terminal, and it will become a background process (d is for detached in the command line).

To access it, you simply need to go to the following url:

`your.device.ip.address:5000`

To find your device's IP address, you can run the `ipconfig` command on your terminal to find it. Any device on the same network can visit this URL to see the playground!

If you wish to then stop and remove the container:

```
docker stop <flask-container>
docker rm <flask-container>
```
To then remove the image itself, you need to run:

`docker image remove <my-flask-app>`