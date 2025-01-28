import os
import yaml
import pandas as pd
import ragas
from ragas import EvaluationDataset
from pathlib import Path
from ragas.metrics import AnswerRelevancy, AnswerCorrectness
from inference import answer_query

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

        answer, _ = answer_query(
            item["user_input"],
            history=[],
            collection_name="deee"
        )
        test_samples.append({
            "user_input": user_input,
            "response": answer,
            "reference": reference
        })

    df = pd.DataFrame(test_samples)
    dataset = EvaluationDataset.from_pandas(df)

    metrics = [
        AnswerRelevancy(),
        AnswerCorrectness()
    ]

    result = ragas.evaluate(dataset, metrics=metrics)


    print("\nRAGAS Evaluation Scores:")
    print(result.to_pandas().mean(numeric_only=True))

    upload_result = result.upload()
    print("RAGAS results can be viewed at:")
    print(upload_result)

if __name__ == "__main__":
    main()