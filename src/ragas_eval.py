import os
import yaml
import pandas as pd
import ragas
from ragas import EvaluationDataset
from pathlib import Path
from ragas.metrics import AnswerRelevancy, AnswerCorrectness, FactualCorrectness, Faithfulness, LLMContextRecall
from inference import answer_query
# import openpyxl   #uncomment this to save the results in Excel format

import dotenv
dotenv.load_dotenv()
RAGAS_APP_TOKEN = os.getenv("RAGAS_APP_TOKEN")

def main():
    # data_file = os.path.join("data", "eval_data.yaml")
    # with open(data_file, "r") as f:
    #     qa_pairs = yaml.safe_load(f)
    current_dir = Path(__file__).parent  # src directory
    data_path = current_dir.parent / "data" / "eval_data.yaml"

    with open(data_path, "r") as f:
        qa_pairs = yaml.safe_load(f)

    test_samples = []
    for item in qa_pairs:
        user_input = item["user_input"]
        reference = item["expected_response"]

        answer, retrieved_context = answer_query(
            item["user_input"],
            history=[],
            collection_name="text_spitter"
        )
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

    print("\nRAGAS Evaluation Scores:")
    print(result.to_pandas().mean(numeric_only=True))

    upload_result = result.upload()
    print("RAGAS results can be viewed at \n:")
    print(upload_result)

    #optional: save the evaluation results to an Excel file
    # metrics_df = result.to_pandas()
    #
    # combined_df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    #
    # excel_file = "evaluation.xlsx"
    # combined_df.to_excel(excel_file, index=False)
    # print(f"Combined evaluation results saved to '{excel_file}'.")

if __name__ == "__main__":
    main()