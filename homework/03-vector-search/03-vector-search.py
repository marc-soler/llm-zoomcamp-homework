# %%
# 1
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-distilbert-cos-v1")

# %%
user_question = "I just discovered the course. Can I still join it?"

# %%
query_vector = model.encode(user_question)
query_vector.shape

# %%
# 2
import json

with open("../../03-vector-search/documents-with-ids.json", "rt") as f:
    documents = json.load(f)

# %%
ml_zoomcamp = [doc for doc in documents if doc["course"] == "machine-learning-zoomcamp"]
len(ml_zoomcamp)

# %%
from tqdm.auto import tqdm
import numpy as np

embeddings = []

for doc in tqdm(ml_zoomcamp):
    qa_text = f"{doc['question']} {doc['text']}"
    qa_text_embed = model.encode(qa_text)
    embeddings.append(qa_text_embed)

# %%
X = np.array(embeddings)
(X.shape)

# %%
# 3
v = query_vector
scores = X.dot(v)

# %%
scores.max()


# %%
# 4
class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


search_engine = VectorSearchEngine(documents=ml_zoomcamp, embeddings=X)
search_engine.search(v, num_results=5)

# %%
import pandas as pd

df_ground_truth = pd.read_csv("../../03-vector-search/ground-truth-data.csv")
df_ground_truth = df_ground_truth[df_ground_truth.course == "machine-learning-zoomcamp"]
ground_truth = df_ground_truth.to_dict(orient="records")

# %%
relevant_results = []

for q in tqdm(ground_truth):
    doc_id = q["document"]
    question_vector = model.encode(q["question"])
    results = search_engine.search(question_vector, num_results=5)
    relevant_results_present = [doc["id"] == doc_id for doc in results]
    relevant_results.append(relevant_results_present)


# %%
def hit_rate(relevant_results):
    cnt = 0
    for line in relevant_results:
        if True in line:
            cnt += 1
    return cnt / len(relevant_results)


# %%
hit_rate(relevant_results)

# %%
# 5
from elasticsearch import Elasticsearch

client = Elasticsearch("http://localhost:9200")

# %%
index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "question": {"type": "text"},
            "section": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
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
for doc, q in zip(ml_zoomcamp, X):
    doc["vector"] = q
    client.index(index=index_name, document=doc)

# %%
query = {
    "field": "vector",
    "query_vector": query_vector,
    "k": 5,
    "num_candidates": 10000,
}

result = client.search(
    index=index_name, knn=query, source=["text", "question", "section", "course", "id"]
)
result["hits"]["hits"]

# %%
# 6
relevant_results_es = []

for q in tqdm(ground_truth):
    doc_id = q["document"]
    question_vector = model.encode(q["question"])
    query = {
        "field": "vector",
        "query_vector": question_vector,
        "k": 5,
        "num_candidates": 10000,
    }
    results = client.search(
        index=index_name,
        knn=query,
        source=["text", "question", "section", "course", "id"],
    )
    results = results["hits"]["hits"]
    relevant_results_present = [doc["_source"]["id"] == doc_id for doc in results]
    relevant_results_es.append(relevant_results_present)

# %%
hit_rate(relevant_results_es)
