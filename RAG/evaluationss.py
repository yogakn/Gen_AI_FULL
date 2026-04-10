import os
os.environ["OPENAI_API_KEY"] = "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"
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

data = {
    "question": ["What causes diabetes?"],
    "answer": ["Diabetes is caused by insulin resistance."],
    "contexts": [[
        "Diabetes occurs when the body becomes resistant to insulin."
    ]],
    "ground_truth": ["Diabetes is caused by insulin resistance."]
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print("\n===== RAG Evaluation =====\n")

scores = result.scores   # this is a LIST

for item in scores:
    for metric, value in item.items():
        val = value if value == value else "N/A"
        print(f"{metric.upper():<20} : {val}")