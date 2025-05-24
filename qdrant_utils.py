from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
import os
import shutil
import numpy as np
from uuid import uuid4
from helpers import UPLOAD_FOLDER

DIM_LENGTH = 384
TABLE_OF_CONTENTS_INDEX = "table_of_contents"

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.local_embedder = SentenceTransformer("intfloat/e5-small-v2")
        self.ensure_upload_folder()
        self.ensure_table_of_contents()

    def ensure_upload_folder(self):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    def get_embedding(self, text):
        text = text.strip()
        return self.local_embedder.encode(text, normalize_embeddings=True).tolist() if text else np.zeros(DIM_LENGTH).tolist()

    def ensure_table_of_contents(self):
        if not self.client.collection_exists(TABLE_OF_CONTENTS_INDEX):
            self.client.create_collection(
                collection_name=TABLE_OF_CONTENTS_INDEX,
                vectors_config=VectorParams(size=DIM_LENGTH, distance=Distance.COSINE)
            )

    def list_collections(self):
        return [c.name for c in self.client.get_collections().collections]

    def list_indexes(self):
        return [c for c in self.list_collections() if c != TABLE_OF_CONTENTS_INDEX]

    def create_index(self, index_name):
        if not self.client.collection_exists(index_name):
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=DIM_LENGTH, distance=Distance.COSINE)
            )
            self.create_topic_directory(index_name)

    def delete_index(self, index_name):
        if self.client.collection_exists(index_name):
            self.client.delete_collection(index_name)
        self.delete_topic_directory(index_name)
        self.client.delete(
            collection_name=TABLE_OF_CONTENTS_INDEX,
            points_selector=Filter(
                must=[FieldCondition(key="index_name", match=MatchValue(value=index_name))]
            )
        )

    def create_topic_directory(self, index_name):
        os.makedirs(os.path.join(UPLOAD_FOLDER, index_name), exist_ok=True)

    def delete_topic_directory(self, index_name):
        shutil.rmtree(os.path.join(UPLOAD_FOLDER, index_name), ignore_errors=True)

    def upsert_metadata(self, index_name, description):
        embedding = self.get_embedding(description)
        self.client.upsert(
            collection_name=TABLE_OF_CONTENTS_INDEX,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "index_name": index_name,
                        "description": description
                    }
                )
            ]
        )

    def get_index_description(self, index_name):
        results = self.client.search(
            collection_name=TABLE_OF_CONTENTS_INDEX,
            query_vector=self.get_embedding(index_name),
            limit=1,
            with_payload=True,
            query_filter=Filter(
                must=[FieldCondition(key="index_name", match=MatchValue(value=index_name))]
            )
        )
        if results and results[0].payload:
            return results[0].payload.get("description", "No description available.")
        return "No description available."

    def get_descriptions(self):
        descriptions = {}
        indexes = self.list_indexes()
        for index in indexes:
            descriptions[index] = self.get_index_description(index)
        return descriptions

    def delete_vectors_by_source(self, index_name, file_name):
        self.client.delete(
            collection_name=index_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=file_name))]
            )
        )

    def is_embedded(self, index_name, file_name):
        try:
            results = self.client.search(
                collection_name=index_name,
                query_vector=np.zeros(DIM_LENGTH).tolist(),
                limit=1,
                with_payload=False,
                query_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=file_name))]
                )
            )
            return len(results) > 0
        except Exception:
            return False

    def upsert_vectors(self, index_name, src_doc, file_paths, chunks, embed_type):
        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=self.get_embedding(chunk),
                    payload={
                        "content": chunk,
                        "source": src_doc,
                        "file_path": file_paths[i],
                        "type": embed_type
                    }
                )
            )
        self.client.upsert(collection_name=index_name, points=points)

    def query_at_index(self, index_name, query, top_k=5, filter=None):
        embedding = self.get_embedding(query)
        query_filter = None
        if filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=val["$eq"]))
                    for key, val in filter.items()
                ]
            )
        results = self.client.search(
            collection_name=index_name,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter
        )
        return [point.payload for point in results]

vector_store_manager = QdrantManager()
