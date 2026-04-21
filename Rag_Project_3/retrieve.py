import numpy as np
from utils import get_embedding

def retrieve(query, index, docs, k=5):
    query_vec = np.array(get_embedding(query)).astype("float32")
    distances, indices = index.search(query_vec, k)

    return [docs[i] for i in indices[0]]