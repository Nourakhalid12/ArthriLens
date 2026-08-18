import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np

class SimpleVectorStore:
    """
    A lightweight, robust local vector store that uses NumPy for cosine similarity search
    and pickles documents, metadata, and embeddings to disk.
    
    This avoids complex database setups and runs instantly for datasets with thousands of items.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds documents and their corresponding pre-computed embeddings.
        
        Args:
            documents: List of dicts, each with 'text' and 'metadata'.
            embeddings: Numpy array of shape (num_docs, embedding_dim).
        """
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents and embeddings must match.")
            
        if len(documents) == 0:
            return

        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def _compute_keyword_scores(self, query_text: str) -> np.ndarray:
        if not query_text or not self.documents:
            return np.zeros(len(self.documents))
            
        import re
        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
            
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return np.zeros(len(self.documents))
            
        # Compute Document Frequencies (DF)
        dfs = {}
        doc_tokens_list = []
        for doc in self.documents:
            tokens = set(tokenize(doc.get("text", "")))
            doc_tokens_list.append(tokens)
            for t in tokens:
                dfs[t] = dfs.get(t, 0) + 1
                
        num_docs = len(self.documents)
        
        # Compute TF-IDF scores
        scores = []
        for tokens in doc_tokens_list:
            score = 0.0
            for token in query_tokens:
                if token in tokens:
                    df = dfs.get(token, 1)
                    idf = np.log(1.0 + num_docs / df)
                    score += idf
            scores.append(score)
            
        scores = np.array(scores)
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score
        return scores

    def search(self, query_embedding: np.ndarray, query_text: str = None, k: int = 5, hybrid_alpha: float = 0.7) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a hybrid search combining Cosine Similarity with a lightweight TF-IDF keyword overlap score.
        
        Args:
            query_embedding: Numpy array of shape (embedding_dim,).
            query_text: The raw query text for keyword matching.
            k: Number of top documents to retrieve.
            hybrid_alpha: Weight of vector similarity (1 - hybrid_alpha weight of keyword matching).
            
        Returns:
            List of tuples (doc, similarity_score) sorted in descending order of similarity.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Ensure query embedding is 1D
        query_embedding = query_embedding.flatten()
        
        # Calculate dot products
        dot_products = np.dot(self.embeddings, query_embedding)
        
        # Calculate magnitudes (norms)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        # Calculate cosine similarity (add epsilon to prevent division by zero)
        similarities = dot_products / (doc_norms * query_norm + 1e-9)
        
        # Hybrid Search: Combine vector similarity with keyword matching
        if query_text is not None:
            keyword_scores = self._compute_keyword_scores(query_text)
            similarities = hybrid_alpha * similarities + (1 - hybrid_alpha) * keyword_scores
        
        # Get the top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results

    def save(self, file_path: str):
        """
        Saves the vector store to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Vector store saved successfully to: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'SimpleVectorStore':
        """
        Loads a vector store from disk.
        """
        store = cls()
        if not os.path.exists(file_path):
            print(f"No existing vector store file found at: {file_path}. Initializing new store.")
            return store
            
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            store.documents = data.get("documents", [])
            store.embeddings = data.get("embeddings", None)
            print(f"Vector store loaded successfully from: {file_path} (containing {len(store.documents)} chunks)")
        except Exception as e:
            print(f"Error loading vector store from {file_path}: {e}. Initializing new store.")
            
        return store
