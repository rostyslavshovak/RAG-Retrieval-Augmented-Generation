# **RAGAS Evaluation**

This document describes how **RAGAS** (Retrieval-Augmented Generation Assessment Score) is applied to this project.

**RAGAS** (Retrieval-Augmented Generation Assessment Score) provides a suite of metrics to evaluate the chatbot’s performance. Key metrics include:

- **Answer Relevance**: Measures how on-topic the answer is relative to the user’s question.
- **Answer Correctness**: Evaluates whether the information in the answer match the expected facts.
- **Factual Correctness**: Checks the alignment of numeric or factual data with ground-truth references.
- **Faithfulness**: Assesses if the chatbot’s generated response accurately reflects (i.e., does not add to or distort) the retrieved text.
- **Context Recall**: Determines whether the system retrieves all necessary information for the user’s question.

**Typical Scores** (from provided evaluations)  
- *Answer Relevance:* ~0.98  
- *Answer Correctness:* ~0.69 
- *Factual Correctness:* ~0.43  
- *Faithfulness:* ~0.95  
- *Context Recall:* ~0.87  

A **higher** score is better. For instance, an *Answer Correctness* of ~0.69 means that while answers are frequently valid, there is room to improve in ensuring numeric or factual precision. The *Factual Correctness* around 0.43 highlights that the chatbot might mix up specific details (like exact figures from the 10-K).

### **Running the Evaluation**
1. Modify or confirm the environment variables in `.env` for your setup.  
2. Run the `ragas_eval.py` script, which uses dataset `data/eval_data.yaml`.
   ```bash
    python -m src.ragas_eval
    ```
3. Inspect the console output for a summary of each metric’s average score.  
4. (Optional) If you have `RAGAS_APP_TOKEN` set, results can be uploaded to the RAGAS Dashboard interface for centralized tracking.

   
### Hypothesis Testing
Take into account that RAGAS is automatically calculated based on the provided evaluation data. If you want to test the hypothesis that the chatbot's performance has improved, you can use the provided evaluation data to perform a statistical test. 
For example, you could use a *ground-truth* (expert or human evaluation) evaluation approach to ensure that the RAGAS scores are consistent with human evaluation.
    - Such approach can be viewed in the `evaluation_testing.xlsx` example file. This is only a demonstration and should be adapted to your specific use cases.