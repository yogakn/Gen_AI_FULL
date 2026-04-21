from retrieve import retrieve
from rerank import rerank
from utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

dataset = [
    {
        "question": " Psycho-Cybernetics?",
        "relevant_docs": ["maxwell"]
    },
    {
        "question": "who is Kuvempu?",
        "relevant_docs": ["K.V. Puttappa (Kuvempu) was born in 1904 in Kuppalli."]
    }
]

def is_relevant(doc, relevant_docs, threshold=0.75):
    emb1 = get_embedding(doc)[0]
    for rel in relevant_docs:
        emb2 = get_embedding(rel)[0]
        sim = cosine_similarity([emb1], [emb2])[0][0]
        if sim >= threshold:
            return True
    return False

def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved if is_relevant(d, relevant)) / k

def recall_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved if is_relevant(d, relevant)) / len(relevant)

def f1(p, r):
    return 0 if p+r==0 else 2*(p*r)/(p+r)

def mrr(retrieved, relevant):
    for i, d in enumerate(retrieved):
        if is_relevant(d, relevant):
            return 1/(i+1)
    return 0

def evaluate(index, docs, k=5):
    results = []
    for item in dataset:
        q = item["question"]
        rel = item["relevant_docs"]

        retrieved = retrieve(q, index, docs, k=10)   # get more
        retrieved = rerank(q, retrieved, top_k=k)   # rerank to K

        p = precision_at_k(retrieved, rel, k)
        r = recall_at_k(retrieved, rel, k)
        f = f1(p, r)
        m = mrr(retrieved, rel)

        results.append({
            "question": q,
            "P@K": round(p,3),
            "R@K": round(r,3),
            "F1": round(f,3),
            "MRR": round(m,3)
        })
    return results
