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
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
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


class GeminiEmbedder(BaseEmbedder):
    """
    Embedder using the Gemini API (REST) for embeddings.
    """
    def __init__(self, model_name: str = None, api_key: str = None):
        from arthrilens.config import GEMINI_API_KEY, EMBEDDING_MODEL
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or EMBEDDING_MODEL
        
        if not self.api_key:
            print("[WARN] No GEMINI_API_KEY found! Using LocalEmbedder as fallback.")
            self.local_fallback = LocalEmbedder()
        else:
            self.local_fallback = None

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.local_fallback:
            return self.local_fallback.embed_documents(texts)
            
        if not texts:
            return np.empty((0, 0))

        import requests
        import time
        all_embeddings = []
        
        for idx, text in enumerate(texts):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
            
            payload = {
                "model": f"models/{self.model_name}",
                "content": {
                    "parts": [{"text": text}]
                }
            }
            
            backoff = 2
            max_retries = 2
            retry_count = 0
            success = False
            embedding_vector = []
            
            while retry_count < max_retries:
                try:
                    response = requests.post(url, json=payload, timeout=20)
                    if response.status_code == 200:
                        res_data = response.json()
                        embedding_vector = res_data.get("embedding", {}).get("values", [])
                        success = True
                        break
                    elif response.status_code == 429:
                        res_text = response.text
                        if "Quota exceeded" in res_text and ("PerDay" in res_text or "daily limit" in res_text.lower()):
                            print("[WARN] Gemini API Daily Quota Exceeded! Automatically falling back to LocalEmbedder offline model.")
                            self.local_fallback = LocalEmbedder()
                            remaining_texts = texts[idx:]
                            remaining_embeddings = self.local_fallback.embed_documents(remaining_texts)
                            if len(all_embeddings) > 0:
                                first_emb = np.array(all_embeddings)
                                if first_emb.shape[1] != remaining_embeddings.shape[1]:
                                    raise RuntimeError(
                                        f"Gemini API rate limit hit and fallback to LocalEmbedder failed due to dimension mismatch. "
                                        f"Expected {first_emb.shape[1]} dimensions from Gemini, but LocalEmbedder returned {remaining_embeddings.shape[1]} dimensions."
                                    )
                                return np.vstack([first_emb, remaining_embeddings])
                            else:
                                return remaining_embeddings
                                
                        print(f"[WARN] Gemini API Rate Limit (429) hit. Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 10)
                        retry_count += 1
                    else:
                        raise ValueError(f"Gemini API returned error {response.status_code}: {response.text}")
                except Exception as e:
                    # Reraise RuntimeErrors from inner blocks
                    if isinstance(e, RuntimeError) and "dimension mismatch" in str(e):
                        raise e
                    print(f"[WARN] Exception during API call: {e}. Retrying...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10)
                    retry_count += 1
            
            if not success:
                print("[WARN] Gemini API persistent rate limits. Automatically falling back to LocalEmbedder offline model.")
                self.local_fallback = LocalEmbedder()
                remaining_texts = texts[idx:]
                remaining_embeddings = self.local_fallback.embed_documents(remaining_texts)
                if len(all_embeddings) > 0:
                    first_emb = np.array(all_embeddings)
                    if first_emb.shape[1] != remaining_embeddings.shape[1]:
                        raise RuntimeError(
                            f"Gemini API rate limit hit and fallback to LocalEmbedder failed due to dimension mismatch. "
                            f"Expected {first_emb.shape[1]} dimensions from Gemini, but LocalEmbedder returned {remaining_embeddings.shape[1]} dimensions."
                        )
                    return np.vstack([first_emb, remaining_embeddings])
                else:
                    return remaining_embeddings
                    
            all_embeddings.append(embedding_vector)
            
            # Print progress
            if (idx + 1) % 10 == 0 or (idx + 1) == len(texts):
                print(f"Embedded {idx + 1}/{len(texts)} chunks...")
                
            # Active pacing
            time.sleep(1.2)
                
        return np.array(all_embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        if self.local_fallback:
            return self.local_fallback.embed_query(text)
            
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        
        payload = {
            "model": f"models/{self.model_name}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code == 200:
                res_data = response.json()
                embedding_vector = res_data.get("embedding", {}).get("values", [])
                return np.array(embedding_vector)
            elif response.status_code == 429:
                print("[WARN] Gemini API Rate Limit (429) hit during query. Falling back to LocalEmbedder offline model.")
                self.local_fallback = LocalEmbedder()
                return self.local_fallback.embed_query(text)
            else:
                raise ValueError(f"Gemini API returned error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[WARN] Exception during API query embedding: {e}. Falling back to LocalEmbedder offline model.")
            self.local_fallback = LocalEmbedder()
            return self.local_fallback.embed_query(text)

