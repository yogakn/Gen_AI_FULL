from utils import client

def generate_answer(query, contexts):
    context = "\n".join(contexts)

    prompt = f"""
Answer ONLY from the context.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content