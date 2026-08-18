import os
import json
import time
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arthrilens.llm import LLMManager

# Global RAG resources
retriever_instance = None
llm_manager = LLMManager()

def get_retriever():
    global retriever_instance
    if retriever_instance is None:
        from arthrilens.retriever import Retriever
        from arthrilens.config import DEFAULT_VECTOR_STORE_PATH
        from arthrilens.embedder import LocalEmbedder
        if not os.path.exists(DEFAULT_VECTOR_STORE_PATH):
            raise FileNotFoundError("Vector store database file does not exist yet. Please run the ingestion pipeline first.")
        retriever_instance = Retriever(vector_store_path=DEFAULT_VECTOR_STORE_PATH)
        if retriever_instance.vector_store.embeddings is not None:
            dim = retriever_instance.vector_store.embeddings.shape[1]
            if dim == 384:
                print("[INFO] Vector store has 384 dimensions. Aligning retriever to LocalEmbedder.")
                retriever_instance.embedder.local_fallback = LocalEmbedder()
    return retriever_instance

class RAGRequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, content: bytes, content_type: str, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        # Handle CORS preflight request
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # Serve frontend assets
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            file_path = os.path.join("frontend", "index.html")
            content_type = "text/html; charset=utf-8"
        elif path == "/index.css":
            file_path = os.path.join("frontend", "index.css")
            content_type = "text/css; charset=utf-8"
        elif path == "/index.js":
            file_path = os.path.join("frontend", "index.js")
            content_type = "application/javascript; charset=utf-8"
        else:
            self._send_response(b"Not Found", "text/plain", 404)
            return

        if not os.path.exists(file_path):
            self._send_response(f"File not found: {file_path}".encode("utf-8"), "text/plain", 404)
            return

        with open(file_path, "rb") as f:
            content = f.read()
        self._send_response(content, content_type)

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                query = data.get("query", "").strip()
                k = int(data.get("k", 5))
                
                if not query:
                    response_bytes = json.dumps({"error": "Query cannot be empty"}).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 400)
                    return
                
                # 1. Retrieval
                try:
                    retriever = get_retriever()
                    results = retriever.retrieve(query, k=k)
                except FileNotFoundError as e:
                    response_bytes = json.dumps({
                        "error": "Vector store database is not built yet! Please run the ingestion pipeline first."
                    }).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 503)
                    return
                
                if not results:
                    response_bytes = json.dumps({
                        "answer": "No relevant documents found in the database. Please try another query.",
                        "references": [],
                        "model": "None",
                        "provider": "None",
                        "latency": 0.0,
                        "logs": [],
                        "evaluation": {
                            "context_relevance": {"score": 0, "explanation": "No documents retrieved"},
                            "faithfulness": {"score": 0, "explanation": "No documents retrieved"},
                            "answer_relevance": {"score": 0, "explanation": "No documents retrieved"}
                        },
                        "avg_similarity": 0.0
                    }).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 200)
                    return
                
                context_chunks = []
                references = []
                scores = []
                
                for doc, score in results:
                    context_chunks.append(doc["text"])
                    scores.append(score)
                    references.append({
                        "text": doc["text"],
                        "score": round(score, 4),
                        "metadata": doc["metadata"]
                    })
                
                avg_similarity = float(sum(scores) / len(scores))
                
                # 2. Generation
                context_str = "\n\n".join([f"Source [{r['metadata'].get('file_name', 'Doc')}]: {r['text']}" for r in references])
                prompt = f"""You are ArthriLens AI, an expert arthritis and joint health clinical assistant.
Your task is to answer the User Question using ONLY the provided Clinical Reference Context.

Strict Instructions:
1. Answer the question accurately using facts directly mentioned in the Context.
2. If the context does not contain the answer or is unrelated, state: "The retrieved clinical guidelines do not contain sufficient information to answer this question."
3. Do not formulate recommendations or facts that are not supported by the retrieved documentation.

Context:
{context_str}

User Question: {query}
"""
                system_instruction = "You are a professional, helpful medical assistant. You answer queries using retrieved clinical documentation only."
                
                generation_res = llm_manager.generate(prompt, system_instruction=system_instruction)
                answer = generation_res.get("text", "")
                
                # 3. Evaluation
                eval_res = llm_manager.evaluate_rag(query, context_chunks, answer)
                
                response_data = {
                    "answer": answer,
                    "references": references,
                    "model": generation_res.get("model", "None"),
                    "provider": generation_res.get("provider", "None"),
                    "latency": generation_res.get("latency", 0.0),
                    "logs": generation_res.get("logs", []),
                    "evaluation": eval_res,
                    "avg_similarity": round(avg_similarity, 4)
                }
                
                response_bytes = json.dumps(response_data).encode("utf-8")
                self._send_response(response_bytes, "application/json", 200)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                response_bytes = json.dumps({"error": f"Internal server error: {str(e)}"}).encode("utf-8")
                self._send_response(response_bytes, "application/json", 500)
                
        elif self.path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                filename = data.get("filename", "").strip()
                file_data = data.get("file_data", "")
                
                if not filename or not file_data:
                    response_bytes = json.dumps({"error": "Filename and file contents cannot be empty"}).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 400)
                    return
                    
                # Clean filename
                filename = os.path.basename(filename)
                ext = os.path.splitext(filename.lower())[1]
                
                if ext not in [".pdf", ".xlsx", ".xls", ".txt"]:
                    response_bytes = json.dumps({"error": f"Unsupported file type: {ext}. Only .pdf, .xlsx, .xls, and .txt are supported."}).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 400)
                    return
                    
                # Base64 decode
                if "," in file_data:
                    file_data = file_data.split(",")[1]
                    
                import base64
                try:
                    file_bytes = base64.b64decode(file_data)
                except Exception as e:
                    response_bytes = json.dumps({"error": f"Invalid Base64 file format: {str(e)}"}).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 400)
                    return
                    
                # Save file
                from arthrilens.config import DEFAULT_DATA_DIR
                uploads_dir = os.path.join(DEFAULT_DATA_DIR, "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                saved_file_path = os.path.join(uploads_dir, filename)
                
                with open(saved_file_path, "wb") as f:
                    f.write(file_bytes)
                    
                # Ingest
                from arthrilens.parser import DocumentParser
                parser = DocumentParser()
                parsed_docs = parser.parse_file(saved_file_path)
                
                if not parsed_docs:
                    if os.path.exists(saved_file_path):
                        os.remove(saved_file_path)
                    response_bytes = json.dumps({"error": "No text content could be extracted from this document."}).encode("utf-8")
                    self._send_response(response_bytes, "application/json", 422)
                    return
                    
                from arthrilens.splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
                chunks = splitter.split_documents(parsed_docs)
                
                from arthrilens.vector_store import SimpleVectorStore
                from arthrilens.config import DEFAULT_VECTOR_STORE_PATH
                vector_store = SimpleVectorStore.load(DEFAULT_VECTOR_STORE_PATH)
                
                from arthrilens.config import EMBEDDING_MODEL
                from arthrilens.embedder import GeminiEmbedder, LocalEmbedder
                embedder = GeminiEmbedder(model_name=EMBEDDING_MODEL)
                
                # Align upload embedder to matching dimension of current database
                if vector_store.embeddings is not None:
                    dim = vector_store.embeddings.shape[1]
                    if dim == 384:
                        print("[INFO] Aligning upload embedder to LocalEmbedder (384 dimensions).")
                        embedder.local_fallback = LocalEmbedder()
                
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = embedder.embed_documents(chunk_texts)
                
                vector_store.add_documents(chunks, embeddings)
                vector_store.save(DEFAULT_VECTOR_STORE_PATH)
                
                # Clear active instance so the next query reloads the store
                global retriever_instance
                retriever_instance = None
                
                response_data = {
                    "success": True,
                    "message": f"Successfully indexed file: {filename}",
                    "chunks_added": len(chunks)
                }
                response_bytes = json.dumps(response_data).encode("utf-8")
                self._send_response(response_bytes, "application/json", 200)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                response_bytes = json.dumps({"error": f"Internal server error: {str(e)}"}).encode("utf-8")
                self._send_response(response_bytes, "application/json", 500)
        else:
            self._send_response(b"Not Found", "text/plain", 404)

def run_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"\n============================================================")
    print(f"ArthriLens Dashboard Server running at: http://localhost:{port}")
    print(f"============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
