# %%
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

# %%
from elasticsearch import Elasticsearch

elastic_client = Elasticsearch("http://localhost:9200")


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


# %%
query = "How do I run kafka?"
response = rag(query)
