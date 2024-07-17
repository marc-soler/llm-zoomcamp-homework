# %%
# 1 - Loading the documents into elasticsearch
# %%
import json

with open("documents-with-ids.json", "rt") as f:
    documents_id = json.load(f)

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# %%
from elasticsearch import Elasticsearch

client = Elasticsearch("http://localhost:9200")

index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "question": {"type": "text"},
            "section": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
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
from tqdm.auto import tqdm

for doc in tqdm(documents_id):
    question = doc["question"]
    text = doc["text"]
    qt = question + " " + text
    doc["question_vector"] = model.encode(question)
    doc["text_vector"] = model.encode(text)
    doc["question_text_vector"] = model.encode(qt)
    client.index(index=index_name, document=doc)


# %%
# 2 - Querying the index
# %%
query = "I just discovered the course. Can I still join it?"

v_q = model.encode(query)

# %%
search_query = {
    "field": "question_vector",
    "query_vector": v_q,
    "k": 5,
    "num_candidates": 10000,
}

res = client.search(
    index=index_name,
    knn=search_query,
    source=["text", "question", "section", "course", "id"],
)
res["hits"]["hits"]

# %%
# 3 - Filtering the results
search_query = {
    "field": "question_vector",
    "query_vector": v_q,
    "k": 5,
    "num_candidates": 10000,
    "filter": {"term": {"course": "data-engineering-zoomcamp"}},
}

res = client.search(
    index=index_name,
    knn=search_query,
    source=["text", "question", "section", "course", "id"],
)

result_docs = []

for hit in res["hits"]["hits"]:
    result_docs.append(hit["_source"])

result_docs


# %%
# 4 - Searching function
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "question", "section", "course", "id"],
    }

    response = client.search(index=index_name, body=search_query)

    results_docs = []

    for hit in response["hits"]["hits"]:
        results_docs.append(hit["_source"])

    return results_docs


# %%
# Test
elastic_search_knn(
    "question_vector",
    v_q,
    "data-engineering-zoomcamp",
)


# %%
# 5 - Wrapper function
def question_vector_knn(q):
    question = q["question"]
    course = q["course"]

    v_q = model.encode(question)

    return elastic_search_knn("question_vector", v_q, course)


# %%
# 6 - Evaluating the results
# %%
import pandas as pd

document_questions = pd.read_csv("ground-truth-data.csv")
ground_truth = document_questions.to_dict(orient="records")


# %%
def hit_rate(relevant_results):
    cnt = 0
    for line in relevant_results:
        if True in line:
            cnt += 1
    return cnt / len(relevant_results)


def mrr(relevant_results):
    cnt = 0
    for line in relevant_results:
        if True in line:
            cnt += 1 / (line.index(True) + 1)
    return cnt / len(relevant_results)


# %%
def evaluate(ground_truth, search_function):
    relevant_results = []

    for q in tqdm(ground_truth):
        doc_id = q["document"]
        results = search_function(q)
        relevant_results_present = [doc["id"] == doc_id for doc in results]
        relevant_results.append(relevant_results_present)

    return {"hit_rate": hit_rate(relevant_results), "mrr": mrr(relevant_results)}


# %%
evaluate(ground_truth, question_vector_knn)
# {'hit_rate': 0.773071104387292, 'mrr': 0.6661407679561998}


# %%
# 7 - Trying with question text vectors
def question_text_vector_knn(q):
    question = q["question"]
    course = q["course"]

    v_q = model.encode(question)

    return elastic_search_knn("question_text_vector", v_q, course)


# %%
evaluate(ground_truth, question_text_vector_knn)
# {'hit_rate': 0.9172249837907932, 'mrr': 0.8237662992579791}

# %%
