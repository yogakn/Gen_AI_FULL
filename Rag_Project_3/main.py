from ingest import ingest
from evaluate import evaluate
from generate import generate_answer

def main():
    index, docs = ingest("data.pdf")

    print("\nRunning evaluation...\n")
    results = evaluate(index, docs, k=5)

    for r in results:
        print("\n---")
        print(r)

    print("\nSample Query:")
    q = "Explain the main topic"
    ans = generate_answer(q, index, docs)
    print(ans)

if __name__ == "__main__":
    main()
