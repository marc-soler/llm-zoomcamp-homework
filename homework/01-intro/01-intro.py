# %%
# !curl localhost:9200

# %%
from elasticsearch import Elasticsearch
import json
from tqdm import tqdm

# %%
with open("../01-intro/documents.json", "rt") as f:
    documents_raw = json.load(f)

documents = []

for course in documents_raw:
    course_name = course["course"]

    for doc in course["documents"]:
        doc["course"] = course_name
        documents.append(doc)

# %%
elastic_client = Elasticsearch("http://localhost:9200")

# %%
index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "question": {"type": "text"},
            "section": {"type": "text"},
            "course": {"type": "keyword"},
        }
    },
}
index_name = "course-questions-homework"

elastic_client.indices.create(index=index_name, body=index_settings)

# %%
for doc in tqdm(documents):
    elastic_client.index(index=index_name, document=doc)

# %%
query = "How do I execute a command in a running docker container?"

# %%
search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "question^4"],
                    "type": "best_fields",
                }
            },
        }
    },
}
response = elastic_client.search(index=index_name, body=search_query)

# %%
search_query_ml = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "question^4"],
                    "type": "best_fields",
                }
            },
            "filter": {"term": {"course": "machine-learning-zoomcamp"}},
        }
    },
}
response_ml = elastic_client.search(index=index_name, body=search_query_ml)
response_ml_docs = [hit["_source"] for hit in response_ml["hits"]["hits"]]

# %%
context = ""

for doc in response_ml_docs:
    context_template = f"""
Q: {doc['question']}
A: {doc['text']}
    """.strip()
    context += context_template + "\n\n"

# %%
prompt_template = f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {query}

CONTEXT:
{context}
""".strip()

len(prompt_template)

# %%
import tiktoken

# %%
encoding = tiktoken.encoding_for_model("gpt-4o")

# %%
len(encoding.encode(prompt_template))

# %%
encoding.decode_single_token_bytes(4165)

# %%
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# %%
gpt_response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": prompt_template}]
)
gpt_response.choices[0].message.content

# %%
input_token_price = 0.005 / 1000
output_token_price = 0.015 / 1000
prompt_price = (
    input_token_price * gpt_response.usage.prompt_tokens
    + output_token_price * gpt_response.usage.completion_tokens
)
