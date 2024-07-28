# RAG_Evaluation


This project implements and evaluates a Retrieval-Augmented Generation (RAG) system for a travel itinerary recommender. The system retrieves relevant contexts for user queries and generates answers using Google Generative AI. The evaluation process ensures that the generated answers are faithful, relevant, and accurate.

**Youtube:** https://www.youtube.com/watch?v=X4Q3vdFc78M&ab_channel=bhargavisikhakolli

**Document:** https://drive.google.com/file/d/1s6gYk4cZpHYz_nmvZcPe6nQA-hHxAiEx/view?usp=sharing


## Prerequisites

Ensure you have the following installed:
- Python 3.7 or later
- The required packages listed in `requirements.txt`

Install the required packages using:

```
pip install -r requirements.txt
```

## Evaluation Metrics
The evaluation process uses the following metrics to assess the quality of the generated answers:

**Faithfulness:** Ensures the answer is supported by the context

**Answer Relevancy:** Measures how relevant the answer is to the query

**Context Precision:** Precision of the context used in the answer

**Context Recall:** Recall of the context used in the answer

**Context Entity Recall:** Recall of entities mentioned in the context

**Answer Similarity:** Similarity between the generated answer and the ground truth answer

**Answer Correctness:** Correctness of the generated answer

**Harmfulness:** Measures potential harmful content in the answer



Before making improvements to the system, the following metrics were evaluated:

- **Faithfulness**: 0.67
- **Answer Relevancy**: 0.65
- **Context Precision**: 0.67
- **Context Recall**: 0.33
- **Context Entity Recall**: 0.67
- **Answer Similarity**: 0.92
- **Answer Correctness**: 0.56
- **Harmfulness**: 0.00

## Improvements Made

Several improvements were made to enhance the performance of the RAG system. These include:

1. **Enhanced Context Retrieval**: Improved the context retrieval process to ensure more relevant contexts are retrieved and reranked effectively.
2. 
3. **Refined Answer Generation**: Updated the prompt and generation method to produce more accurate and contextually appropriate answers.

## Improved Evaluation Scores

After implementing the improvements, the system was re-evaluated and the scores showed significant enhancements in key metrics:

- **Answer Relevancy**: Improved from 0.65 to 0.83
- **Answer Correctness**: Improved from 0.56 to 0.68




### The main script RAG_improvements.py includes the following steps:

**Environment Setup:** Load API keys and initialize the Google Generative AI client.

**Data Preparation:** Define user queries, ground truth contexts, and ground truth answers.

**Context Retrieval:** Use BM25 to retrieve the most relevant contexts for each query.

**Context Reranking:** Rerank the retrieved contexts based on simple relevance scoring.

**Answer Generation:** Generate answers using Google Generative AI.

**Post-Processing:** Apply rule-based corrections to the generated answers.

**Evaluation:** Evaluate the generated answers using various metrics.


## Summary of Improvements

The improvements in the RAG system led to a notable increase in the relevancy and correctness of the answers generated. The metrics reflect a more reliable and user-friendly system capable of providing better travel itinerary recommendations.

The focus on enhancing context retrieval and refining answer generation has proven effective in achieving these improvements, demonstrating the importance of these components in the overall performance of the RAG system.


## Evaluation Scope:

### Feedback Loop:

Benefits of a Feedback Loop in a Recommender System

**Performance Monitoring:** It helps in monitoring the performance of the recommendations in real-time. If users frequently provide negative feedback (e.g., not following through with the recommendations), it signals that the model needs adjustments.

**User Preferences:** It allows the system to learn from user behavior and preferences. Positive feedback (e.g., users acting on recommendations) can be used to reinforce similar recommendations in the future.

**Dynamic Updates:** The model can be updated dynamically based on the latest feedback, ensuring that it stays relevant and accurate over time.

**Personalization:** Continuous feedback helps in fine-tuning the recommendations to better match individual user preferences, leading to more personalized experiences.

**Error Correction:** If the model makes errors, feedback helps in identifying and correcting these errors, leading to more accurate recommendations.

### How to Implement a Feedback Loop

**Collect Feedback:** Implement mechanisms for users to provide feedback on the recommendations. This could be explicit (e.g., thumbs up/down, ratings) or implicit (e.g., clicks, purchases).

**Analyze Feedback:** Regularly analyze the collected feedback to identify patterns and trends. Determine what types of recommendations are working well and which ones are not.

**Update Model:** Incorporate the feedback into the training process to improve the accuracy and relevance of future recommendations.

**A/B Testing:** Continuously run A/B tests to compare the performance of different versions of the model. Use the feedback to choose the best-performing version.

**Iterate:** Continuously iterate on the model based on feedback. The goal is to create a cycle of continuous improvement.

