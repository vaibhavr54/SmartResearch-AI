import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


class Memory:
    def __init__(self):

        self.docs = []
        self.vecs = []
        self.index = None

    def add(self, texts):
        for text in texts:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                v = embed_model.encode(chunk)
                v = v / np.linalg.norm(v)
                self.docs.append(chunk)
                self.vecs.append(v)

        if self.vecs:
            dim = len(self.vecs[0])
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.array(self.vecs))

    def search(self, q):
        if not self.index:
            return []

        v = embed_model.encode(q)
        v = v / np.linalg.norm(v)

        _, I = self.index.search(np.array([v]), 5)
        return [self.docs[i] for i in I[0]]
