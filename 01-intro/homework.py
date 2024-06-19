# %%
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
client = OpenAI(api_key=OPENAI_API_KEY)

# %%
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Is it too late to joing the course?"}],
)

# %%
response.choices[0].message.content
