# %%
import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch, BadRequestError
import json

# %%
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
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

try:
    elastic_client.indices.create(index=index_name, body=index_settings)

    with open("../01-intro/documents.json", "rt") as f:
        docs = json.load(f)

    documents = []

    for course_dict in docs:
        for doc in course_dict["documents"]:
            doc["course"] = course_dict["course"]
            documents.append(doc)

    for doc in documents:
        elastic_client.index(index=index_name, document=doc)

except BadRequestError:
    pass


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
    response = elastic_client.search(index="course-questions", body=search_query)
    result_docs = [hit["_source"] for hit in response["hits"]["hits"]]

    return result_docs


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
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def rag(query):
    context = elastic_search(query)
    prompt = build_prompt(query, context)
    answer = llm(prompt)
    return answer


def main():
    st.title("Course content RAG")

    # Input box
    user_input = st.text_input("Enter your question:")

    # Button to invoke the rag function
    if st.button("Ask"):
        with st.spinner("Processing..."):
            output = rag(user_input)
            st.success("Done!")
            st.write(output)


# %%
if __name__ == "__main__":
    main()
