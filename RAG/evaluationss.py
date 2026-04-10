from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Sample dataset
data = {
    "question": [
        "What causes diabetes?"
    ],
    "answer": [
        "Diabetes is caused by insulin resistance."
    ],
    "contexts": [[
        "Diabetes occurs when the body becomes resistant to insulin."
    ]],
    "ground_truth": [
        "Diabetes is caused by insulin resistance."
    ]
}

dataset = Dataset.from_dict(data)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)