# %%
import dlt
from rest_api import RESTAPIConfig, rest_api_source
from dlt.sources.helpers.rest_client.paginators import (
    BasePaginator,
    JSONResponsePaginator,
)
from dlt.sources.helpers.requests import Response, Request
from dlt.destinations.adapters import lancedb_adapter
import lancedb
import ollama
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["DESTINATION__LANCEDB__EMBEDDING_MODEL_PROVIDER"] = "sentence-transformers"
os.environ["DESTINATION__LANCEDB__EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["DESTINATION__LANCEDB__CREDENTIALS__URI"] = ".lancedb"


# %%


class PostBodyPaginator(BasePaginator):
    def __init__(self):
        super().__init__()
        self.cursor = None

    def update_state(self, response: Response) -> None:
        # Assuming the API returns an empty list when no more data is available
        if not response.json():
            self._has_next_page = False
        else:
            self.cursor = response.json().get("next_cursor")
            if self.cursor is None:
                self._has_next_page = False

    def update_request(self, request: Request) -> None:
        if request.json is None:
            request.json = {}

        # Add the cursor to the request body
        request.json["start_cursor"] = self.cursor


@dlt.resource(name="homework")
def rest_api_notion_resource():
    notion_config: RESTAPIConfig = {
        "client": {
            "base_url": "https://api.notion.com/v1/",
            "auth": {"token": dlt.secrets["sources.rest_api.notion.api_key"]},
            "headers": {
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            },
        },
        "resources": [
            {
                "name": "search",
                "endpoint": {
                    "path": "search",
                    "method": "POST",
                    "paginator": PostBodyPaginator(),
                    "json": {
                        "query": "homework",
                        "sort": {
                            "direction": "ascending",
                            "timestamp": "last_edited_time",
                        },
                    },
                    "data_selector": "results",
                },
            },
            {
                "name": "page_content",
                "endpoint": {
                    "path": "blocks/{page_id}/children",
                    "paginator": JSONResponsePaginator(),
                    "params": {
                        "page_id": {
                            "type": "resolve",
                            "resource": "search",
                            "field": "id",
                        }
                    },
                },
            },
        ],
    }

    yield from rest_api_source(notion_config, name="homework")


def extract_page_content(response):
    block_id = response["id"]
    last_edited_time = response["last_edited_time"]
    block_type = response.get("type", "Not paragraph")
    if block_type != "paragraph":
        content = ""
    else:
        try:
            content = response["paragraph"]["rich_text"][0]["plain_text"]
        except IndexError:
            content = ""
    return {
        "block_id": block_id,
        "block_type": block_type,
        "content": content,
        "last_edited_time": last_edited_time,
        "inserted_at_time": datetime.now(timezone.utc),
    }


@dlt.resource(
    name="homework",
    write_disposition="merge",
    primary_key="block_id",
    columns={"last_edited_time": {"dedup_sort": "desc"}},
)
def rest_api_notion_incremental(
    last_edited_time=dlt.sources.incremental(
        "last_edited_time",
        initial_value="2024-06-26T08:16:00.000Z",
        primary_key=("block_id"),
    )
):
    # last_value = last_edited_time.last_value
    # print(last_value)

    for block in rest_api_notion_resource.add_map(extract_page_content):
        if not (len(block["content"])):
            continue
        yield block


def load_notion() -> None:
    pipeline = dlt.pipeline(
        pipeline_name="homework",
        destination="lancedb",
        dataset_name="notion_pages",
        # full_refresh=True
    )

    load_info = pipeline.run(
        lancedb_adapter(rest_api_notion_incremental, embed="content"),
        table_name="homework",
        write_disposition="merge",
    )
    print(load_info)


load_notion()

# %%
db = lancedb.connect(".lancedb")
dbtable = db.open_table("notion_pages___homework")

dbtable.to_pandas().last_edited_time.max()


# %%
def retrieve_context_from_lancedb(dbtable, question, top_k=2):

    query_results = dbtable.search(query=question).to_list()
    context = "\n".join([result["content"] for result in query_results[:top_k]])

    return context


def main():
    # Connect to the lancedb table
    db = lancedb.connect(".lancedb")
    dbtable = db.open_table("notion_pages___homework")

    # A system prompt telling ollama to accept input in the form of "Question: ... ; Context: ..."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps users understand policies inside a company's employee handbook. The user will first ask you a question and then provide you relevant paragraphs from the handbook as context. Please answer the question based on the provided context. For any details missing in the paragraph, encourage the employee to contact the HR for that information. Please keep the responses conversational.",
        }
    ]

    # Accept user question
    question = "how many PTO days are the employees entitled to in a year?"

    # Retrieve the relevant paragraphs on the question
    context = retrieve_context_from_lancedb(dbtable, question, top_k=2)

    # Create a user prompt using the question and retrieved context
    messages.append(
        {"role": "user", "content": f"Question: '{question}'; Context:'{context}'"}
    )

    # Get the response from the LLM
    response = ollama.chat(model="llama2-uncensored", messages=messages)
    response_content = response["message"]["content"]
    print(f"Assistant: {response_content}")

    # Add the response into the context window
    messages.append({"role": "assistant", "content": response_content})


main()
