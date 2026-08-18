import os
import time
import json
import requests
from typing import Dict, Any, List, Tuple

class LLMManager:
    """
    Manages generation using a fallback sequence of free-tier LLM providers:
    1. Gemini (Primary)
    2. Groq (Secondary)
    3. OpenRouter (Tertiary)
    
    Also provides unified RAG Triad evaluation.
    """
    def __init__(self):
        from arthrilens.config import GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_MODEL, GROQ_MODEL, OPENROUTER_MODEL
        self.gemini_key = GEMINI_API_KEY
        self.groq_key = GROQ_API_KEY
        self.openrouter_key = OPENROUTER_API_KEY
        
        self.gemini_model = GEMINI_MODEL
        self.groq_model = GROQ_MODEL
        self.openrouter_model = OPENROUTER_MODEL

    def generate(self, prompt: str, system_instruction: str = None) -> Dict[str, Any]:
        """
        Attempts to generate text by going through the fallback sequence.
        Returns a dictionary with the generated text, active model, latency, and logs.
        """
        logs = []
        start_total = time.time()
        
        # 1. Try Gemini
        if self.gemini_key:
            logs.append({"provider": "Gemini", "status": "trying", "model": self.gemini_model})
            start_api = time.time()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
                
            try:
                response = requests.post(url, json=payload, timeout=20)
                latency = time.time() - start_api
                if response.status_code == 200:
                    res_data = response.json()
                    # Extract text
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        logs[-1].update({"status": "success", "latency": round(latency, 2)})
                        return {
                            "text": text,
                            "model": self.gemini_model,
                            "provider": "Gemini",
                            "latency": round(time.time() - start_total, 2),
                            "logs": logs
                        }
                    else:
                        raise ValueError("No generation candidates returned from Gemini.")
                else:
                    logs[-1].update({"status": "failed", "error": f"HTTP {response.status_code}: {response.text[:200]}"})
            except Exception as e:
                logs[-1].update({"status": "failed", "error": str(e)})
        else:
            logs.append({"provider": "Gemini", "status": "skipped", "error": "No API key configured"})

        # 2. Try Groq
        if self.groq_key:
            logs.append({"provider": "Groq", "status": "trying", "model": self.groq_model})
            start_api = time.time()
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.groq_model,
                "messages": messages
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=20)
                latency = time.time() - start_api
                if response.status_code == 200:
                    res_data = response.json()
                    text = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logs[-1].update({"status": "success", "latency": round(latency, 2)})
                    return {
                        "text": text,
                        "model": self.groq_model,
                        "provider": "Groq",
                        "latency": round(time.time() - start_total, 2),
                        "logs": logs
                    }
                else:
                    logs[-1].update({"status": "failed", "error": f"HTTP {response.status_code}: {response.text[:200]}"})
            except Exception as e:
                logs[-1].update({"status": "failed", "error": str(e)})
        else:
            logs.append({"provider": "Groq", "status": "skipped", "error": "No API key configured"})

        # 3. Try OpenRouter
        if self.openrouter_key:
            logs.append({"provider": "OpenRouter", "status": "trying", "model": self.openrouter_model})
            start_api = time.time()
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Nourakhalid12/ArthriLens",
                "X-Title": "ArthriLens RAG"
            }
            
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.openrouter_model,
                "messages": messages
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=20)
                latency = time.time() - start_api
                if response.status_code == 200:
                    res_data = response.json()
                    text = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logs[-1].update({"status": "success", "latency": round(latency, 2)})
                    return {
                        "text": text,
                        "model": self.openrouter_model,
                        "provider": "OpenRouter",
                        "latency": round(time.time() - start_total, 2),
                        "logs": logs
                    }
                else:
                    logs[-1].update({"status": "failed", "error": f"HTTP {response.status_code}: {response.text[:200]}"})
            except Exception as e:
                logs[-1].update({"status": "failed", "error": str(e)})
        else:
            logs.append({"provider": "OpenRouter", "status": "skipped", "error": "No API key configured"})

        # If everything fails
        return {
            "text": "Error: All configured LLM providers failed to generate content or no keys were provided. Please check your API keys or quotas.",
            "model": "None",
            "provider": "None",
            "latency": round(time.time() - start_total, 2),
            "logs": logs
        }

    def evaluate_rag(self, query: str, context_chunks: List[str], answer: str) -> Dict[str, Any]:
        """
        Runs RAG Triad Evaluation in a single LLM request to save API quota and minimize latency.
        Evaluates Context Relevance, Faithfulness (Groundedness), and Answer Relevance.
        """
        context_str = "\n---\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""You are an expert AI RAG evaluator. Your task is to evaluate a Retrieval-Augmented Generation (RAG) system response.
You are given the user's query, the retrieved context chunks (which the system used as a reference), and the generated answer.

Analyze and score the following three metrics (from 0 to 100):
1. context_relevance: How relevant are the retrieved context chunks to the user's query? (0% means completely irrelevant, 100% means perfect context for answering the query).
2. faithfulness: Is the generated answer fully grounded in the retrieved context? It should NOT contain information or claims that are unsupported by the context (0% means completely hallucinated/unsupported, 100% means every claim is fully supported by the context).
3. answer_relevance: Does the generated answer directly address the user's query? (0% means it answers something completely different, 100% means it is a precise, complete, and direct answer to the query).

Query: {query}
Retrieved Context Chunks: {context_str}
Generated Answer: {answer}

Provide your feedback in strict JSON format. Do not add any markdown formatting, explanation, or other text outside the JSON.
Response schema:
{{
  "context_relevance": {{
    "score": <int>,
    "explanation": "<str>"
  }},
  "faithfulness": {{
    "score": <int>,
    "explanation": "<str>"
  }},
  "answer_relevance": {{
    "score": <int>,
    "explanation": "<str>"
  }}
}}"""

        system_instruction = "You are a strict, objective AI evaluation judge. You output ONLY valid JSON matching the requested schema."
        
        eval_res = self.generate(prompt, system_instruction=system_instruction)
        text = eval_res.get("text", "").strip()
        
        # Clean markdown code block wraps if LLM outputted them
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Defaults if parsing fails
        default_eval = {
            "context_relevance": {"score": 0, "explanation": "Failed to parse evaluator response"},
            "faithfulness": {"score": 0, "explanation": "Failed to parse evaluator response"},
            "answer_relevance": {"score": 0, "explanation": "Failed to parse evaluator response"}
        }
        
        try:
            eval_dict = json.loads(text)
            # Validate structure
            for metric in ["context_relevance", "faithfulness", "answer_relevance"]:
                if metric not in eval_dict or "score" not in eval_dict[metric] or "explanation" not in eval_dict[metric]:
                    raise ValueError(f"Missing key in metric {metric}")
            return eval_dict
        except Exception as e:
            print(f"[WARN] Failed to parse evaluator response: {e}. Output was:\n{text}")
            return default_eval
