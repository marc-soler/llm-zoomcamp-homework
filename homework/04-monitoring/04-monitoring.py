# %%
# LOADING DATA
import pandas as pd

df = pd.read_csv("../../04-monitoring/data/results-gpt4o-mini.csv")
df = df.iloc[:300]

# %%
# EMBEDDING MODEL
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# %%
# EMBEDDING THE FIRST ANSWER
answer_llm = df.iloc[0].answer_llm
embedding_model.encode(answer_llm)

# %%
# DOT PRODUCT A -> A'
from tqdm.auto import tqdm

evaluations = []

for _, row in tqdm(df).iterrows():
    embedded_answer_orig = embedding_model.encode(row.answer_orig)
    embedded_answer_llm = embedding_model.encode(row.answer_llm)
    evaluations.append(embedded_answer_orig.dot(embedded_answer_llm))

evaluations_df = pd.DataFrame(evaluations)
evaluations_df.describe()

# %%
# COSINE SIMILARITY A -> A'
import numpy as np


def normalise_vector(v):
    norm = np.sqrt((v * v).sum())
    v_norm = v / norm
    return v_norm


evaluations_norm = []

for _, row in tqdm(df.iterrows()):
    embedded_answer_orig = embedding_model.encode(row.answer_orig)
    embedded_answer_llm = embedding_model.encode(row.answer_llm)
    embedded_answer_orig_norm = normalise_vector(embedded_answer_orig)
    embedded_answer_llm_norm = normalise_vector(embedded_answer_llm)
    evaluations_norm.append(embedded_answer_orig_norm.dot(embedded_answer_llm_norm))

evaluations_norm_df = pd.DataFrame(evaluations_norm)
evaluations_norm_df.describe()

# %%
# ROUGE SCORE
from rouge import Rouge

rouge_scorer = Rouge()

r = df.iloc[10]
scores = rouge_scorer.get_scores(r["answer_llm"], r["answer_orig"])[0]
print(scores)

# %%
# AVERAGE F-SCORE FROM ROUGE
np.mean([scores["f"] for metric, scores in scores.items()])

# %%
# AVERAGE ROUGE 2 F-SCORE
f_scores = []

for _, row in tqdm(df.iterrows()):
    scores = rouge_scorer.get_scores(row["answer_llm"], row["answer_orig"])[0]
    f_scores.append(scores["rouge-2"]["f"])

np.mean(f_scores)
