# %%
# 1 - Preparing the documents
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
# 2 - Assigning IDs to the documents based on course + question
import hashlib


def generate_document_id(doc):
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    doc_id = hash_hex[:8]
    return doc_id


for doc in documents:
    doc["id"] = generate_document_id(doc)

# %%
## Checking for duplicates in the hashes
from collections import defaultdict

hashes = defaultdict(list)

for doc in documents:
    hashes[doc["id"]].append(doc)

len(hashes), len(documents)

for k, values in hashes.items():
    if len(values) > 1:
        print(k, values)

# %%
# 3 - Saving the documents with IDs
with open("documents_with_ids.json", "wt") as f_out:
    json.dump(documents, f_out)

# !head documents_with_ids.json

# %%
# 4 - Generating questions from the documents
# %%
# 4.1 Prompt template
prompt_template = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record. 

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()

# %%
# 4.2 Creating a prompt in OpenAI
from openai import OpenAI

client = OpenAI()


def generate_questions(doc):
    prompt = prompt_template.format(**doc)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    json_response = response.choices[0].message.content
    return json_response


# %%
# 4.3 Generating the questions
from tqdm.auto import tqdm

results = {}

for doc in tqdm(documents):
    doc_id = doc["id"]
    if doc_id in results:
        continue
    # questions = generate_questions(doc)
    # results[doc_id] = questions

# Let's not run this for loop, as it costs ~4$

# %%
# 5 - Final result
import pandas as pd

document_questions = pd.read_csv("ground-truth-data.csv")
