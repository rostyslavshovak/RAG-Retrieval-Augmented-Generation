import os
import yaml
import pandas as pd
import ragas
from ragas import EvaluationDataset

from ragas.metrics import (
    AnswerRelevancy,
    AnswerCorrectness,
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
)

from src.inference import answer_query
import dotenv
dotenv.load_dotenv()
RAGAS_APP_TOKEN = os.getenv("RAGAS_APP_TOKEN")

def main():
    data_file = os.path.join("../data", "eval_data.yaml")
    with open(data_file, "r", encoding="utf-8") as f:
        qa_list = yaml.safe_load(f)

    test_samples = []
    for item in qa_list:
        user_input = item["user_input"]
        reference = item["expected_response"]

        answer, retrieved_context = answer_query(user_input, history=[], collection_name="merged_context")

        test_samples.append({
            "user_input": user_input,
            "response": answer,
            "retrieved_contexts": [retrieved_context],
            "reference": reference
        })

    df = pd.DataFrame(test_samples)

    expected_columns = {"user_input", "response", "retrieved_contexts", "reference"}
    if not expected_columns.issubset(df.columns):
        missing = expected_columns - set(df.columns)
        print(f"Error: Missing columns in DataFrame: {missing}")
        return

    dataset = EvaluationDataset.from_pandas(df)

    metrics = [
        AnswerRelevancy(),
        AnswerCorrectness(),
        FactualCorrectness(),
        Faithfulness(),
        LLMContextRecall()
    ]

    result = ragas.evaluate(dataset, metrics=metrics)

    print("     RAGAS EVALUATION SCORES     ")
    if hasattr(result, "results"):
        for metric_name, val in result.results.items():
            print(f"{metric_name}: {val:.4f}")
    else:
        print(result)

    upload_result = result.upload()
    print("RAGAS results can be viewed at:")
    print(upload_result)

if __name__ == "__main__":
    main()