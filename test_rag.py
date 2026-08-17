import os
import sys
import numpy as np

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arthrilens.parser import DocumentParser
from arthrilens.splitter import RecursiveCharacterTextSplitter
from arthrilens.embedder import LocalEmbedder
from arthrilens.vector_store import SimpleVectorStore

def test_parser_and_splitter():
    print("Testing Parser & Splitter...")
    parser = DocumentParser()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    
    # Try parsing the first available PDF in clinical guidelines or supplements
    sample_pdf = os.path.join("arthrilens", "data", "nccih", "supplements.pdf")
    if os.path.exists(sample_pdf):
        print(f"Parsing sample PDF: {sample_pdf}")
        docs = parser.parse_pdf(sample_pdf)
        print(f"[OK] Parsed PDF: {len(docs)} pages.")
        if docs:
            chunks = splitter.split_documents(docs[:1])
            print(f"[OK] Chunked page 1 into {len(chunks)} chunks.")
            print(f"Sample chunk: {chunks[0]['text'][:100]}...")
    else:
        print(f"[WARN] PDF test skipped (file not found: {sample_pdf})")

    # Try parsing Excel
    sample_excel = os.path.join("arthrilens", "data", "nccih", "mind_body.xlsx")
    if os.path.exists(sample_excel):
        print(f"Parsing sample Excel: {sample_excel}")
        docs = parser.parse_excel(sample_excel)
        print(f"[OK] Parsed Excel: {len(docs)} rows.")
        if docs:
            chunks = splitter.split_documents(docs[:5])
            print(f"[OK] Chunked top 5 rows into {len(chunks)} chunks.")
            print(f"Sample chunk: {chunks[0]['text'][:100]}...")
    else:
        print(f"[WARN] Excel test skipped (file not found: {sample_excel})")

def test_vector_store():
    print("\nTesting Vector Store & Cosine Similarity search...")
    store = SimpleVectorStore()
    
    # Mock data
    mock_docs = [
        {"text": "Rheumatoid arthritis is an autoimmune condition that causes joint pain and inflammation.", "metadata": {"source": "mock1.pdf"}},
        {"text": "A healthy diet and regular low-impact exercise can improve overall joint health.", "metadata": {"source": "mock2.pdf"}},
        {"text": "EULAR guidelines recommend early diagnosis and DMARD therapy for RA patients.", "metadata": {"source": "mock3.pdf"}}
    ]
    
    # Mock embeddings (dimension 4)
    # Using specific vectors to verify cosine similarity
    mock_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Close to query [1, 0, 0, 0]
        [0.0, 1.0, 0.0, 0.0],
        [0.8, 0.1, 0.0, 0.0]   # Also close to query [1, 0, 0, 0]
    ])
    
    store.add_documents(mock_docs, mock_embeddings)
    
    # Query vector
    query_vector = np.array([1.0, 0.0, 0.0, 0.0])
    results = store.search(query_vector, k=2)
    
    print(f"[OK] Search returned {len(results)} results.")
    for idx, (doc, score) in enumerate(results):
        print(f"  Result {idx+1}: Score={score:.4f} | Text='{doc['text']}'")
        
    assert len(results) == 2, "Should return 2 results"
    assert results[0][0]["metadata"]["source"] == "mock1.pdf", "Best result should be mock1.pdf"
    print("[OK] Vector Store unit test passed!")

if __name__ == "__main__":
    print("Running unit tests for RAG pipeline component correctness...\n")
    test_parser_and_splitter()
    test_vector_store()
    print("\nAll unit tests passed successfully!")
