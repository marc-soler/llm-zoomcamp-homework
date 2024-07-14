# %%
# 1 - Preparing the documents
# %%
import json

with open("../01-intro/documents.json", "rt") as f:
    docs_raw = json.load(f)

documents = []
# Flattening the JSON structure
for course in docs_raw:
    for doc in course["documents"]:
        doc["course"] = course["course"]
        documents.append(doc)

# %%
# 2 - Creating embeddings using pretrained models
# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

# %%
model.encode("This is a sentence").shape

# %%
embeddings = []
for doc in documents:
    doc["vector"] = model.encode(doc["text"]).tolist()
    embeddings.append(doc)


# embeddings = model.encode([doc["text"] for doc in documents])

# %%
# 3 - Connecting to Elasticsearch
# %%
from elasticsearch import Elasticsearch
from tqdm import tqdm

client = Elasticsearch("http://localhost:9200")

# %%
# 4 - Creating a Mapping and Index
# %%
index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "question": {"type": "text"},
            "section": {"type": "text"},
            "course": {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}
index_name = "course-questions"

client.indices.delete(index=index_name, ignore_unavailable=True)
client.indices.create(index=index_name, body=index_settings)

# %%
# 5 - Indexing the documents
# %%
for doc in tqdm(embeddings):
    try:
        client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)

# %%
# 6 - Creating a user query
# %%
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)

# %%
query = {
    "field": "vector",
    "query_vector": vector_search_term,
    "k": 5,  # Nearest neighbors
    "num_candidates": 10000,  # How many documents to consider in the querys
}

# %%
result = client.search(
    index=index_name, knn=query, source=["text", "question", "section", "course"]
)
result["hits"]["hits"]

# %%
# 7 - Keyword Search
result_advanced = client.search(
    index=index_name,
    query={
        "bool": {
            "must": {
                "multi_match": {
                    "query": search_term,
                    "fields": ["text", "question", "course", "title"],
                    "type": "best_fields",
                }
            },
            "filter": {"term": {"course": "data-engineering-zoomcamp"}},
        }
    },
)
result_advanced["hits"]["hits"]

# %%
# Vector search with filtering
result_filtering = client.search(
    index=index_name,
    query={"match": {"course": "data-engineering-zoomcamp"}},
    knn=query,
    size=5,
    explain=True,
)
result_filtering["hits"]["hits"]
