# %%
# !wget https://raw.githubusercontent.com/marc-soler/llm-zoomcamp/main/01-intro/minsearch.py
# !wget https://raw.githubusercontent.com/marc-soler/llm-zoomcamp/main/01-intro/documents.json

# %%
import minsearch
import json

# %%
with open("documents.json", "rt") as f:
    docs = json.load(f)

# %%
documents = []

for course_dict in docs:
    for doc in course_dict["documents"]:
        doc["course"] = course_dict["course"]
        documents.append(doc)

# %%
index = minsearch.Index(
    text_fields=["question", "text", "section"], keyword_fields=["course"]
).fit(documents)

# %%
index.fit(documents)

# %%
q = "Until when can I join the course?"

boost = {"question": 3.0, "section": 0.5}
results = index.search(
    query=q,
    filter_dict={"course": "data-engineering-zoomcamp"},
    boost_dict=boost,
    num_results=5,
)

# %%
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# %%
context = ""

for doc in results:
    context += f"section: {doc['course']}\nquestion:{doc['question']}\nanswer: {doc['text']}\n\n"

q = "The course has already started, can I still enroll?"

# %%
prompt_template = f"""
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Only use infomation from the CONTEXT when answering the QUESTION.
If the CONTEXT does not contain the answer, output NONE.

QUESTION: {q}

CONTEXT: 
{context}
"""

# %%
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt_template}],
)

response.choices[0].message.content


# %%
# FLOW CLEANING
# %%
def search(query):
    boost = {"question": 3.0, "section": 0.5}
    results = index.search(
        query=query,
        filter_dict={"course": "data-engineering-zoomcamp"},
        boost_dict=boost,
        num_results=5,
    )
    return results


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context += f"section: {doc['course']}\nquestion:{doc['question']}\nanswer: {doc['text']}\n\n"

    prompt_template = f"""
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Only use infomation from the CONTEXT when answering the QUESTION.
        If the CONTEXT does not contain the answer, output NONE.

        QUESTION: {query}

        CONTEXT: 
        {context}
    """.strip()
    return prompt_template


def llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def rag(query):
    context = search(query)
    prompt = build_prompt(query, context)
    answer = llm(prompt)
    return answer


# %%
query = "How do I run kafka?"
rag(query)

# %%
# USING ELASTICSEARCH
# %%
from elasticsearch import Elasticsearch
from tqdm import tqdm

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
index_name = "course-questions"

elastic_client.indices.create(index=index_name, body=index_settings)

# %%
for doc in tqdm(documents):
    elastic_client.index(index=index_name, document=doc)

# %%
query = "How do I run kafka?"


# %%
def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "question^3", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": "data-engineering-zoomcamp"}},
            }
        },
    }
    response = elastic_client.search(index=index_name, body=search_query)
    result_docs = [hit["_source"] for hit in response["hits"]["hits"]]

    return result_docs


def rag(query):
    context = elastic_search(query)
    prompt = build_prompt(query, context)
    answer = llm(prompt)
    return answer


# %%
rag(query)
