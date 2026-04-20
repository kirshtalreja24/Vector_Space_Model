import math
import numpy as np


class Queries:

    def __init__(self, processor):
        self.processor = processor
        self.index = processor.words

        self.all_docs = sorted(
            list({doc_id for postings in self.index.values() for doc_id in postings})
        )

        self.N = len(self.all_docs)
        self.terms = sorted(self.index.keys())
        self.doc_vectors = self.build_doc_vectors()

    def build_doc_vectors(self):
        doc_vectors = []

        for doc_id in self.all_docs:
            vec = []

            for term in self.terms:
                postings = self.index[term]

                tf = len(postings[doc_id]) if doc_id in postings else 0

                # scaling TF
                if tf > 0:
                    tf = 1 + math.log10(tf)

                df = len(postings)

                idf = math.log10((self.N + 1) / (df + 1)) + 1

                vec.append(tf * idf)

            doc_vectors.append(np.array(vec, dtype=float))

        return doc_vectors

    def build_query_vector(self, terms):
        tf_map = {}

        for t in terms:
            tf_map[t] = tf_map.get(t, 0) + 1

        vec = []

        for term in self.terms:
            tf = tf_map.get(term, 0)

            if tf > 0:
                tf = 1 + math.log10(tf)

                postings = self.index.get(term, {})
                df = len(postings)

                idf = math.log10((self.N + 1) / (df + 1)) + 1

                vec.append(tf * idf)
            else:
                vec.append(0.0)

        return np.array(vec, dtype=float)

    def cosine(self, q, d):
        dot = np.dot(q, d)

        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)

        if norm_q == 0 or norm_d == 0:
            return 0.0

        return dot / (norm_q * norm_d)

    def process_query(self, query):
        terms = self.processor.processQuery(query)

        if not terms:
            print(f"Query: {query}\n\nLength=0\nset()")
            return set()

        q_vec = self.build_query_vector(terms)

        if np.linalg.norm(q_vec) == 0:
            print("Query terms not found in vocabulary.")
            return set()

        # Compute cosine similarity scores
        scores = np.array([
            self.cosine(q_vec, d_vec) for d_vec in self.doc_vectors
        ])

        # Rank documents (descending)
        ranked_indices = np.argsort(-scores)

        # Alpha threshold 
        alpha = 0.005
        max_score = np.max(scores)

        result_docs = []

        if max_score == 0:
            print(f"Query: {query}\n\nLength=0\nset()")
            return set()

        for idx in ranked_indices:
            if scores[idx] >= alpha * max_score:
                result_docs.append(str(self.all_docs[idx]))
            else:
                break  
        result_set = set(result_docs)

        print(f"Query: {query}")
        print(f"\nLength={len(result_set)}")
        print(result_set)

        return result_set