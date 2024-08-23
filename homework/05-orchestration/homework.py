# %%
# DATA LOADING
import io
import requests
import docx


def clean_line(line):
    line = line.strip()
    line = line.strip("\uFEFF")
    return line


def read_faq(file_id):
    url = f"https://docs.google.com/document/d/{file_id}/export?format=docx"

    response = requests.get(url)
    response.raise_for_status()

    with io.BytesIO(response.content) as f_in:
        doc = docx.Document(f_in)

    questions = []

    question_heading_style = "heading 2"
    section_heading_style = "heading 1"

    heading_id = ""
    section_title = ""
    question_title = ""
    answer_text_so_far = ""

    for p in doc.paragraphs:
        style = p.style.name.lower()
        p_text = clean_line(p.text)

        if len(p_text) == 0:
            continue

        if style == section_heading_style:
            section_title = p_text
            continue

        if style == question_heading_style:
            answer_text_so_far = answer_text_so_far.strip()
            if (
                answer_text_so_far != ""
                and section_title != ""
                and question_title != ""
            ):
                questions.append(
                    {
                        "text": answer_text_so_far,
                        "section": section_title,
                        "question": question_title,
                    }
                )
                answer_text_so_far = ""

            question_title = p_text
            continue

        answer_text_so_far += "\n" + p_text

    answer_text_so_far = answer_text_so_far.strip()
    if answer_text_so_far != "" and section_title != "" and question_title != "":
        questions.append(
            {
                "text": answer_text_so_far,
                "section": section_title,
                "question": question_title,
            }
        )

    return questions


def load_data(*args, **kwargs):
    documents = []
    faq_documents = {
        "llm-faq_v1": "1T3MdwUvqCL3jrh3d3VCXQ8xE0UqRzI3bfgpfBq3ZWG0",
    }

    for course, file_id in faq_documents.items():
        course_documents = read_faq(file_id)
        documents.append({"course": course, "documents": course_documents})

    return documents


docs = load_data()

# %%
# CHUNKING
import re
from typing import Any, Dict, List


def chunk_documents(data: List[Dict[str, Any]], *args, **kwargs):
    documents = []

    for idx, item in enumerate(data):
        course = item["course"]

        for info in item["documents"]:
            section = info["section"]
            question = info["question"]
            answer = info["text"]

            # Generate a unique document ID
            document_id = ":".join(
                [re.sub(r"\W", "_", part) for part in [course, section, question]]
            ).lower()

            # Format the document string
            chunk = "\n".join(
                [
                    f"course:\n{course}\n",
                    f"section:\n{section}\n",
                    f"question:\n{question}\n",
                    f"answer:\n{answer}\n",
                ]
            )

            documents.append(
                dict(
                    chunk=chunk,
                    document=info,
                    document_id=document_id,
                )
            )

    print(f"Documents:", len(documents))

    return documents


chunked_docs = chunk_documents(docs)

# %%
# EXPORTING
from typing import Dict, List, Tuple, Union
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch


def elasticsearch(
    documents: List[Dict[str, Union[Dict, List[int], np.ndarray, str]]],
    *args,
    **kwargs,
):
    """
    Exports document data to an Elasticsearch database.
    """

    index_name_prefix = kwargs.get("index_name", "documents")
    current_time = datetime.now().strftime("%Y%m%d_%M%S")
    index_name = f"{index_name_prefix}_{current_time}"
    print("index name:", index_name)
    number_of_shards = kwargs.get("number_of_shards", 1)
    number_of_replicas = kwargs.get("number_of_replicas", 0)
    vector_column_name = kwargs.get("vector_column_name", "embedding")

    dimensions = kwargs.get("dimensions")
    if dimensions is None and len(documents) > 0:
        document = documents[0]
        dimensions = len(document.get(vector_column_name) or [])

    es_client = Elasticsearch("http://localhost:9200")

    print("Connecting to Elasticsearch at http://localhost:9200")

    index_settings = {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": number_of_replicas,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "document_id": {"type": "keyword"},
            }
        },
    }

    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        print("Index created with properties:", index_settings)
        print("Embedding dimensions:", dimensions)

    print(f"Indexing {len(documents)} documents to Elasticsearch index {index_name}")
    for document in documents:
        print(f'Indexing document {document["document_id"]}')

        es_client.index(index=index_name, document=document)

    print(document)


elasticsearch(chunked_docs)

# %%
# TESTING THE RETRIEVAL
question = "When is the next cohort?"
index_name = "documents_20240824_4535"


def elastic_search(query, index_name):
    es_client = Elasticsearch("http://localhost:9200")
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        # "fields": ["text", "question^3", "section"],
                        "type": "best_fields",
                    }
                },
            }
        },
    }
    response = es_client.search(index=index_name, body=search_query)
    result_docs = [hit["_source"] for hit in response["hits"]["hits"]]

    return result_docs


elastic_search(question, index_name)
