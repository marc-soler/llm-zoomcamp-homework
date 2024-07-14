# %%
# 1 - Loading the documents into elasticsearch
from elasticsearch import Elasticsearch
import json

client = Elasticsearch("http://localhost:9200")

with open("documents-with-ids.json", "rt") as f:
    documents_id = json.load(f)

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
        }
    },
}
index_name = "course-questions"

client.indices.delete(index=index_name, ignore_unavailable=True)
client.indices.create(index=index_name, body=index_settings)

# %%
from tqdm.auto import tqdm

for doc in tqdm(documents_id):
    client.index(index=index_name, document=doc)


# %%
# 2 - Defining the search function
def elastic_search(query, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": course}},
            }
        },
    }
    response = client.search(index="course-questions", body=search_query)

    results_docs = []

    for hit in response["hits"]["hits"]:
        results_docs.append(hit["_source"])

    return results_docs


# %%
# Test
elastic_search(query="How do I run kafka?", course="data-engineering-zoomcamp")

# %%
# 3 - Searching all the generated questions
import pandas as pd

document_questions = pd.read_csv("ground-truth-data.csv")
ground_truth = document_questions.to_dict(orient="records")

# %%
relevant_results = []

for q in tqdm(ground_truth):
    doc_id = q["document"]
    results = elastic_search(query=q["question"], course=q["course"])
    relevant_results_present = [doc["id"] == doc_id for doc in results]
    relevant_results.append(relevant_results_present)


# %%
# 4 - Calculating Hit Rate (Recall)
def hit_rate(relevant_results):
    cnt = 0
    for line in relevant_results:
        if True in line:
            cnt += 1
    return cnt / len(relevant_results)


# %%
# 5 - Calculating MRR (MRR)
def mrr(relevant_results):
    cnt = 0
    for line in relevant_results:
        if True in line:
            cnt += 1 / (line.index(True) + 1)
    return cnt / len(relevant_results)


# %%
# 6 - Metrics for ElasticSearch
hit_rate(relevant_results), mrr(relevant_results)

# %%
# 7 - Metrics for Minsearch
import sys

sys.path.insert(0, "../01-intro")

import minsearch

# %%
index = minsearch.Index(
    text_fields=["question", "text", "section"], keyword_fields=["course", "id"]
)
index.fit(documents_id)


# %%
def minsearch_search(query, course):
    boost = {"question": 3.0, "section": 0.5}

    results = index.search(
        query=query,
        filter_dict={"course": course},
        boost_dict=boost,
        num_results=5,
    )
    return results


# %%
relevant_results_minsearch = []

for q in tqdm(ground_truth):
    doc_id = q["document"]
    results = minsearch_search(query=q["question"], course=q["course"])
    relevant_results_present = [doc["id"] == doc_id for doc in results]
    relevant_results_minsearch.append(relevant_results_present)

# %%
hit_rate(relevant_results_minsearch), mrr(relevant_results_minsearch)

# %%
# Minsearch seems to be slightly better
