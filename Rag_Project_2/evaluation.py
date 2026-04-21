import os
from ingestion import load_pdf
from preprocessing import clean_text
from chunking import chunk_text
from vector_store import create_collection, insert_chunks
from retriever import retrieve
from rag_pipeline import generate_answer
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

def main():
    # Load and prepare the RAG system
    text = load_pdf("data/hr_policy.pdf")
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    create_collection()
    insert_chunks(chunks)

    # Define test questions and ground truths
    test_questions = [
        "What is the maternity leave policy?",
        "How many days of annual leave are employees entitled to?",
        "What is the procedure for reporting harassment?"
    ]

    ground_truths = [
        "Employees are entitled to 12 weeks of paid maternity leave.",
        "Employees receive 25 days of annual leave per year.",
        "Report harassment to HR within 24 hours."
    ]

    # Collect data for evaluation
    questions = []
    answers = []
    contexts = []
    gt_answers = []

    for q, gt in zip(test_questions, ground_truths):
        retrieved_contexts = retrieve(q)
        answer = generate_answer(q, retrieved_contexts)

        print(f"\nQuestion: {q}")
        print(f"Retrieved Contexts: {retrieved_contexts}")
        print(f"Generated Answer: {answer}")
        print(f"Ground Truth: {gt}")

        questions.append(q)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        gt_answers.append(gt)

    # Create dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": gt_answers
    }

    dataset = Dataset.from_dict(data)

    # Evaluate
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    print("\n===== RAG Evaluation Results =====\n")
    print(result)

if __name__ == "__main__":
    main()