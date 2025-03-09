# **RAGAS Evaluation**

This document outlines how **RAGAS** (Retrieval-Augmented Generation Assessment Score) s utilized in this project.

**RAGAS** (Retrieval-Augmented Generation Assessment Score) provides a suite of metrics to evaluate the chatbot’s performance. Key metrics include:

- **Answer Relevance**: Measures how closely the answer addresses the user's question.
- **Answer Correctness**: Accuracy of the provided information. Evaluates whether the information in the answer match the expected facts.
- **Factual Correctness**: Alignment of facts or numeric data with ground truth.
- **Faithfulness**: Accuracy of responses relative to the retrieved information
- **Context Recall**: Completeness in retrieving necessary information for the user’s question.

**Typical Scores** 
- *Answer Relevance:* ~0.97  
- *Answer Correctness:* ~0.95 
- *Factual Correctness:* ~0.80  
- *Faithfulness:* ~0.94  
- *Context Recall:* ~0.98  

A **higher** score is better. 

For instance, *Factual Correctness* around 0.80 indicates possible inaccuracies in specific details, so the chatbot might mix up specific details (like exact figures from the 10-K).

---
### **Running the Evaluation**
1. Configure environment variables in `.env`.
2. Adjust the `PROMPT_TEMPLATE` and collection [name](https://github.com/rostyslavshovak/RAG-Retrieval-Augmented-Generation/blob/main/src/ragas_eval.py#L33) as needed.
2. Run evaluation:
   ```bash
    python -m src.ragas_eval
    ```
3. Review console output for metric averages.  
- (Optional) Set `RAGAS_APP_TOKEN` to upload evaluation results to the RAGAS Dashboard for centralized tracking.
     - Example (outdated, before dataset changes): View evaluation results for our dataset in RAGAS Dashboard by this [link](https://app.ragas.io/public-shared/alignment/evaluation/a44ac9ac-92ed-4389-b3f1-b3850a0aabf2)

---
### Hypothesis Testing

RAGAS scores are calculated automatically from provided evaluation data. To verify chatbot performance improvements statistically, consider comparing automated RAGAS scores with *ground-truth human evaluations*. 

See the [evaluation_results.xlsx](https://github.com/rostyslavshovak/RAG-Retrieval-Augmented-Generation/blob/main/data/evaluation_results.xlsx) example for a demonstration.

---