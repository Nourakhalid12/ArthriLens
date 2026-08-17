from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class BaseEmbedder:
    """
    Base class for RAG text embedders.
    """
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of documents. Returns a numpy array of shape (num_docs, embedding_dim).
        """
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embeds a single query string. Returns a numpy array of shape (embedding_dim,).
        """
        raise NotImplementedError

class LocalEmbedder(BaseEmbedder):
    """
    Local embedder utilizing PyTorch and Hugging Face Transformers directly.
    Bypasses sentence-transformers to avoid heavy dependencies (like scipy/scikit-learn)
    which may be blocked by system-level security policies.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # standard sentence-transformers repository naming on Hugging Face
        if "/" not in model_name:
            self.hf_model_path = f"sentence-transformers/{model_name}"
        else:
            self.hf_model_path = model_name
            
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.hf_model_path)
            self._model.eval() # Put model in evaluation mode
        return self._model

    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings, accounting for padding.
        """
        token_embeddings = model_output[0] # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of documents in batches to avoid memory overhead.
        """
        if not texts:
            return np.empty((0, 0))
            
        batch_size = 128
        all_embeddings = []
        
        # Disable gradient computation for inference speed
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                # Tokenize sentences
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                )
                # Compute token embeddings
                model_output = self.model(**encoded_input)
                # Perform mean pooling
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                # Normalize embeddings to unit vectors (for cosine similarity match)
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embeds a single query string.
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(
                [text], 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            )
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu().numpy()[0]
