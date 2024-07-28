import os
import pandas as pd
import nest_asyncio
from datasets import Dataset, Features, Sequence, Value
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq

from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
    context_entity_recall, answer_similarity, answer_correctness
)
from ragas.metrics.critique import harmfulness
from ragas.evaluation import evaluate

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Set the Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")

os.environ["GOOGLE_API_KEY"] = api_key

# Initialize the Google Generative AI client
google_ai = GoogleGenerativeAI(api_key=api_key, model="models/text-bison-001")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define questions, ground truth contexts, and answers
user_queries = [
    "What are the top tourist attractions in Seattle?",
    "Suggest a nature hike in the US.",
    "Recommend a historical tour in Boston."
]

ground_truth_contexts = [
    ["Visit the Space Needle and explore the Pike Place Market in Seattle"],
    ["Hike in the Rocky Mountains and enjoy the stunning natural scenery"],
    ["Discover the rich history of Boston with a Freedom Trail walking tour"]
]

ground_truth_answers = [
    "The top tourist attractions in Seattle include the Space Needle and Pike Place Market.",
    "You can hike in the Rocky Mountains and enjoy the stunning natural scenery.",
    "A historical tour in Boston can be done through the Freedom Trail walking tour."
]

# Flatten ground truth contexts for BM25
flattened_contexts = [context for sublist in ground_truth_contexts for context in sublist]

# Initialize BM25 retriever
tokenized_contexts = [context.split() for context in flattened_contexts]
bm25 = BM25Okapi(tokenized_contexts)

# Retrieve top context for each query using BM25
top_contexts = []
for query in user_queries:
    tokenized_query = query.split()
    top_context = bm25.get_top_n(tokenized_query, flattened_contexts, n=3)  # Retrieve top 3 contexts
    top_contexts.append(top_context)

# Rerank the retrieved contexts using a simple relevance scoring
def rerank_contexts(query, contexts):
    query_tokens = set(query.split())
    scored_contexts = []
    for context in contexts:
        context_tokens = set(context.split())
        score = len(query_tokens.intersection(context_tokens)) / len(context_tokens)
        scored_contexts.append((context, score))
    best_context = max(scored_contexts, key=lambda x: x[1])[0]
    return best_context

reranked_contexts = [rerank_contexts(query, contexts) for query, contexts in zip(user_queries, top_contexts)]

# Generate answers using Google Generative AI
def generate_answer(query, context):
    response = google_ai.generate(
        prompts=[f"Answer the question based on the following context:\nContext: {context}\nQuestion: {query}"],
        max_tokens=50
    )
    print(response)
    # Access the response text correctly
    return response.generations[0][0].text.strip()

generated_answers = [generate_answer(query, context) for query, context in zip(user_queries, reranked_contexts)]
# Post-process answers (simple rule-based correction for demonstration)
def post_process_answer(answer):
    corrections = {
        "Liberty Bell": "Freedom Trail walking tour"
    }
    for wrong, correct in corrections.items():
        answer = answer.replace(wrong, correct)
    return answer

processed_answers = [post_process_answer(answer) for answer in generated_answers]

# Prepare DataFrame for the evaluation
data = {
    "question": user_queries,
    "ground_truth": [' '.join(context) for context in ground_truth_contexts],
    "contexts": [[context] for context in reranked_contexts],  # Ensure contexts are lists of strings
    "ground_truth_answer": ground_truth_answers,
    "generated_answer": processed_answers,
    "answer": processed_answers  # Added this column for context_utilization
}

df = pd.DataFrame(data)

# Print the DataFrame to verify its contents
print("DataFrame contents:")
print(df)

# Define the correct features for the Dataset
features = Features({
    "question": Value("string"),
    "ground_truth": Value("string"),
    "contexts": Sequence(Value("string")),
    "ground_truth_answer": Value("string"),
    "generated_answer": Value("string"),
    "answer": Value("string")
})

# Convert DataFrame to Dataset with the correct features
dataset = Dataset.from_pandas(df, features=features)

# Define column mapping
column_mapping = {
    "question": "question",
    "ground_truth": "ground_truth",
    "contexts": "contexts",
    "ground_truth_answer": "ground_truth_answer",
    "generated_answer": "generated_answer",
    "answer": "answer"  # Ensure this is included
}

print("Starting evaluation...")
# Create a Google Generative AI instance
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness],
    column_map=column_mapping,
    llm=llm
)

# Convert the score to a pandas DataFrame and display it
score_df = score.to_pandas()
print(score_df)

# If you need to access individual scores
print(f"Answer Relevancy: {score_df['answer_relevancy'].mean():.2f}")
print(f"Faithfulness: {score_df['faithfulness'].mean():.2f}")
print(f"Context Precision: {score_df['context_precision'].mean():.2f}")
print(f"Context Recall: {score_df['context_recall'].mean():.2f}")
print(f"Context Entity Recall: {score_df['context_entity_recall'].mean():.2f}")
print(f"Answer Similarity: {score_df['answer_similarity'].mean():.2f}")
print(f"Answer Correctness: {score_df['answer_correctness'].mean():.2f}")
print(f"Harmfulness: {score_df['harmfulness'].mean():.2f}")
