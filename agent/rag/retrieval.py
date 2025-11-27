import os
import glob
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class DocumentRetriever:
    def __init__(self, docs_dir: str, chunk_size: int = 256):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        try:
            self.documents = self._load_and_chunk_docs()
            self.corpus = [doc["content"] for doc in self.documents]
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
            else:
                self.bm25 = None
        except Exception as e:
            print(f"Critical Error during DocumentRetriever initialization: {e}")
            self.documents = []
            self.corpus = []
            self.bm25 = None

    def _load_and_chunk_docs(self) -> List[Dict[str, Any]]:
        """Loads all markdown files, chunks them, and assigns unique IDs."""
        all_docs = []
        doc_files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        if not doc_files:
            print(f"Warning: No markdown files found in {self.docs_dir}")
        
        for file_path in doc_files:
            filename = os.path.basename(file_path).replace(".md", "")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            # Simple chunking by splitting on double newline (paragraph)
            chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}::chunk{i}"
                all_docs.append({
                    "id": doc_id,
                    "content": chunk,
                    "source": filename
                })
        return all_docs

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant document chunks using BM25."""
        if self.bm25 is None:
            print("Warning: BM25 not initialized. Returning empty list.")
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": scores[i]
            })
            
        return results

if __name__ == '__main__':
    # Example usage for testing
    # Assuming the script is run from the project root (ai-assignment-dspy)
    retriever = DocumentRetriever("/home/ubuntu/ai-assignment-dspy/docs")
    
    print("--- Test Retrieval (Policy) ---")
    query = "What is the return window for unopened Beverages?"
    results = retriever.retrieve(query, k=2)
    for res in results:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}")
        print(f"Content: {res['content'][:50]}...")
        
    print("\n--- Test Retrieval (KPI) ---")
    query = "What is the formula for Average Order Value?"
    results = retriever.retrieve(query, k=1)
    for res in results:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}")
        print(f"Content: {res['content'][:50]}...")
