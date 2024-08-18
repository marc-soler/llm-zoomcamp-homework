# %%
# Load documents with IDs

# %%
import requests

base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/documents-with-ids.json"
docs_url = f"{base_url}/{relative_url}?raw=1"
docs_response = requests.get(docs_url)
documents = docs_response.json()

# %%
documents[10]

# %%
# Load ground truth

# %%
import pandas as pd

base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
relative_url = "03-vector-search/eval/ground-truth-data.csv"
ground_truth_url = f"{base_url}/{relative_url}?raw=1"

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == "machine-learning-zoomcamp"]
ground_truth = df_ground_truth.to_dict(orient="records")

# %%
ground_truth[10]

# %%
doc_idx = {d["id"]: d for d in documents}
doc_idx["5170565b"]["text"]

# %%
# Index data

# %%
from sentence_transformers import SentenceTransformer

model_name = "multi-qa-MiniLM-L6-cos-v1"
model = SentenceTransformer(model_name)

# %%
from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://localhost:9200")

index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
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

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# %%
from tqdm.auto import tqdm

for doc in tqdm(documents):
    question = doc["question"]
    text = doc["text"]
    doc["question_text_vector"] = model.encode(question + " " + text)

    es_client.index(index=index_name, document=doc)


# %%
# Retrieval
# %%
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
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in es_results["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


def question_text_vector_knn(q):
    question = q["question"]
    course = q["course"]

    v_q = model.encode(question)

    return elastic_search_knn("question_text_vector", v_q, course)


# %%
question_text_vector_knn(
    dict(
        question="Are sessions recorded if I miss one?",
        course="machine-learning-zoomcamp",
    )
)


# %%
# The RAG flow
# %%
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""

    for doc in search_results:
        context = (
            context
            + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        )

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# %%
from openai import OpenAI

client = OpenAI()


def llm(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# %%
# previously: rag(query: str) -> str
def rag(query: dict, model="gpt-4o") -> str:
    search_results = question_text_vector_knn(query)
    prompt = build_prompt(query["question"], search_results)
    answer = llm(prompt, model=model)
    return answer


# %%
ground_truth[10]

# %%
rag(ground_truth[10])

# %%
doc_idx["5170565b"]["text"]

# %%
# Cosine similarity metric

# %%
answer_orig = "Yes, sessions are recorded if you miss one. Everything is recorded, allowing you to catch up on any missed content. Additionally, you can ask questions in advance for office hours and have them addressed during the live stream. You can also ask questions in Slack."
answer_llm = "Everything is recorded, so you won’t miss anything. You will be able to ask your questions for office hours in advance and we will cover them during the live stream. Also, you can always ask questions in Slack."

v_llm = model.encode(answer_llm)
v_orig = model.encode(answer_orig)

v_llm.dot(v_orig)

# %%
ground_truth[0]

# %%
len(ground_truth)

# %%
answers = {}

for i, rec in enumerate(tqdm(ground_truth)):
    if i in answers:
        continue

    answer_llm = rag(rec)
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc["text"]

    answers[i] = {
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
        "question": rec["question"],
        "course": rec["course"],
    }

# %%
results_gpt4o = [None] * len(ground_truth)

for i, val in answers.items():
    results_gpt4o[i] = val.copy()
    results_gpt4o[i].update(ground_truth[i])

# %%
import pandas as pd

# %%
df_gpt4o = pd.DataFrame(results_gpt4o)

# %%
# !mkdir data

# %%
df_gpt4o.to_csv("data/results-gpt4o.csv", index=False)

# %%
# Evaluating GPT 3.5

# %%
rag(ground_truth[10], model="gpt-3.5-turbo")

# %%
# USING MULTITHREADING

from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(max_workers=6)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results


# %%
def process_record(rec):
    model = "gpt-3.5-turbo"
    answer_llm = rag(rec, model=model)

    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc["text"]

    return {
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
        "question": rec["question"],
        "course": rec["course"],
    }


# %%
process_record(ground_truth[10])

# %%
results_gpt35 = map_progress(pool, ground_truth, process_record)

# %%
df_gpt35 = pd.DataFrame(results_gpt35)
df_gpt35.to_csv("data/results-gpt35.csv", index=False)

# %%
# !head data/results-gpt35.csv

# %%
# Cosine similarity

# A->Q->A' cosine similarity

# A -> Q -> A'

# cosine(A, A')

# %%
# gpt-4o

results_gpt4o = pd.read_csv("data/results-gpt4o.csv").to_dict(orient="records")
record = results_gpt4o[0]


# %%
def compute_similarity(record):
    answer_orig = record["answer_orig"]
    answer_llm = record["answer_llm"]

    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)

    return v_llm.dot(v_orig)


# %%
similarity = []

for record in tqdm(results_gpt4o):
    sim = compute_similarity(record)
    similarity.append(sim)

# %%
df_gpt4o = pd.DataFrame(results_gpt4o)
df_gpt4o["cosine"] = similarity
df_gpt4o["cosine"].describe()

# %%
# gpt-3.5-turbo
results_gpt35 = pd.read_csv("data/results-gpt35.csv").to_dict(orient="records")

similarity_35 = []

for record in tqdm(results_gpt35):
    sim = compute_similarity(record)
    similarity_35.append(sim)

# %%
df_gpt35 = pd.DataFrame(results_gpt35)
df_gpt35["cosine"] = similarity_35
df_gpt35["cosine"].describe()


# %%
# gpt-4o-mini
def process_record_4o_mini(rec):
    model = "gpt-4o-mini"
    answer_llm = rag(rec, model=model)

    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc["text"]

    return {
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
        "question": rec["question"],
        "course": rec["course"],
    }


# %%
process_record_4o_mini(ground_truth[10])

# %%
results_gpt4omini = []

# %%
for record in tqdm(ground_truth):
    result = process_record_4o_mini(record)
    results_gpt4omini.append(result)

# %%
df_gpt4o_mini = pd.DataFrame(results_gpt4omini)
df_gpt4o_mini.to_csv("data/results-gpt4o-mini.csv", index=False)

# %%
results_gpt4omini = pd.read_csv("data/results-gpt4o-mini.csv").to_dict(orient="records")

similarity_4o_mini = []

for record in tqdm(results_gpt4omini):
    sim = compute_similarity(record)
    similarity_4o_mini.append(sim)

# %%
df_gpt4o_mini = pd.DataFrame(results_gpt4omini)
df_gpt4o_mini["cosine"] = similarity_4o_mini
df_gpt4o_mini["cosine"].describe()

# %%
import plotly.figure_factory as ff
import numpy as np

hist_data = [df_gpt35["cosine"], df_gpt4o["cosine"], df_gpt4o_mini["cosine"]]
group_labels = ["GPT-3.5", "GPT-4o", "GPT-4o-mini"]

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

fig.update_layout(
    title_text="RAG LLM performance", xaxis_title="A -> Q -> A' cosine similarity"
)
fig.show()

# %%
# LLM-as-a-Judge

# %%
prompt1_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

# %%
df_sample = df_gpt4o_mini.sample(n=150, random_state=1)

# %%
samples = df_sample.to_dict(orient="records")

# %%
record = samples[0]
record

# %%
prompt = prompt1_template.format(**record)
print(prompt)

# %%
answer = llm(prompt, model="gpt-4o-mini")

# %%
import json

# %%
evaluations = []

for record in tqdm(samples):
    prompt = prompt1_template.format(**record)
    evaluation = llm(prompt, model="gpt-4o-mini")
    evaluations.append(evaluation)

# %%
json_evaluations = []

for i, str_eval in enumerate(evaluations):
    json_eval = json.loads(str_eval)
    json_evaluations.append(json_eval)

# %%
df_evaluations = pd.DataFrame(json_evaluations)

# %%
df_evaluations.Relevance.value_counts()

# %%
df_evaluations[df_evaluations.Relevance == "NON_RELEVANT"]  # .to_dict(orient='records')

# %%
sample[4]

# %%
prompt = prompt2_template.format(**record)
print(prompt)

# %%
evaluation = llm(prompt, model="gpt-4o-mini")
print(evaluation)

# %%
evaluations_2 = []

for record in tqdm(samples):
    prompt = prompt2_template.format(**record)
    evaluation = llm(prompt, model="gpt-4o-mini")
    evaluations_2.append(evaluation)

# %%
json_evaluations_2 = []

for i, str_eval in enumerate(evaluations_2):
    json_eval = json.loads(str_eval)
    json_evaluations_2.append(json_eval)

# %%
df_evaluations_2 = pd.DataFrame(json_evaluations_2)

# %%
df_evaluations_2[df_evaluations_2.Relevance == "NON_RELEVANT"]

# %%
samples[45]

# %% [markdown]
# ## Saving all the data

# %%
df_gpt4o.to_csv("data/results-gpt4o-cosine.csv", index=False)
df_gpt35.to_csv("data/results-gpt35-cosine.csv", index=False)
df_gpt4o_mini.to_csv("data/results-gpt4o-mini-cosine.csv", index=False)

# %%
df_evaluations.to_csv("data/evaluations-aqa.csv", index=False)
df_evaluations_2.to_csv("data/evaluations-qa.csv", index=False)
