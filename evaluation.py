import os
import openai
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas.metrics.critique import harmfulness
from ragas.evaluation import evaluate
from datasets import Dataset
import pandas as pd
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")
openai.api_key = api_key

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

retrieved_contexts = [
    ["Visit the Space Needle and explore the Pike Place Market in Seattle"],
    ["Hike in the Rocky Mountains and enjoy the stunning natural scenery"],
    ["Visit the Liberty Bell and Independence Hall in Philadelphia"]
]

ground_truth_answers = [
    "The top tourist attractions in Seattle include the Space Needle and Pike Place Market.",
    "You can hike in the Rocky Mountains and enjoy the stunning natural scenery.",
    "A historical tour in Boston can be done through the Freedom Trail walking tour."
]

generated_answers = [
    "The top tourist attractions in Seattle are the Space Needle and Pike Place Market.",
    "A great nature hike in the US is in the Rocky Mountains.",
    "A historical tour in Boston can be done through the Liberty Bell."
]

# Prepare DataFrame for the evaluation
data = {
    "question": user_queries,
    "ground_truth": [' '.join(context) for context in ground_truth_contexts],
    "contexts": retrieved_contexts,  # Keep this as a list of strings
    "ground_truth_answer": ground_truth_answers,
    "generated_answer": generated_answers,
    "answer": generated_answers  # Added this column for context_utilization
}

df = pd.DataFrame(data)

# Print the DataFrame to verify its contents
print("DataFrame contents:")
print(df)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Define column mapping
column_mapping = {
    "question": "question",
    "ground_truth": "ground_truth",
    "contexts": "contexts",
    "ground_truth_answer": "ground_truth_answer",
    "generated_answer": "generated_answer",
    "answer": "answer"  # Ensure this is included
}

# Evaluate using multiple metrics
print("Starting evaluation...")
score = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness],
    column_map=column_mapping
)

# Convert the score to a pandas DataFrame and display it
score_df = score.to_pandas()
print(score_df)

# If you need to access individual scores
print(f"Faithfulness: {score_df['faithfulness'].mean():.2f}")
print(f"Answer Relevancy: {score_df['answer_relevancy'].mean():.2f}")
print(f"Context Precision: {score_df['context_precision'].mean():.2f}")
print(f"Context Recall: {score_df['context_recall'].mean():.2f}")
print(f"Context Entity Recall: {score_df['context_entity_recall'].mean():.2f}")
print(f"Answer Similarity: {score_df['answer_similarity'].mean():.2f}")
print(f"Answer Correctness: {score_df['answer_correctness'].mean():.2f}")
print(f"Harmfulness: {score_df['harmfulness'].mean():.2f}")
