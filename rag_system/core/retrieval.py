import os
import re
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# Cache path for flashrank model to avoid system directories
RERANK_CACHE = r"C:\Users\26503\Documents\antigravity\magical-hertz\rag_system\flashrank_cache"
os.makedirs(RERANK_CACHE, exist_ok=True)

# Lazy-loaded Flashrank Ranker
_ranker_instance = None

def get_flashrank_ranker():
    """Lazy load the Flashrank Ranker model."""
    global _ranker_instance
    if _ranker_instance is None:
        # Uses ms-marco-MiniLM-L-12-v2 which is highly accurate (~30MB download, extremely fast)
        _ranker_instance = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2", 
            cache_dir=RERANK_CACHE
        )
    return _ranker_instance

def expand_query(query: str, llm) -> list[str]:
    """
    Use the LLM to translate/expand the query into 3 alternative search queries 
    to capture different terminology and improve keyword/vector recall.
    """
    prompt = f"""You are a RAG Query Optimizer. Given a user query, generate 3 alternative variations of the query that target the same meaning but use different terminology, focusing on GIS concepts or tech synonyms.

Original query: "{query}"

Output exactly 3 lines, one query per line, with no labels, numbers, or bullet points.
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise search query generator. Output only the queries, one per line."),
            HumanMessage(content=prompt)
        ])
        lines = response.content.strip().split("\n")
        expanded = [query]
        for line in lines:
            line_clean = re.sub(r"^(\d+\.|\*|-)\s*", "", line).strip()
            if line_clean:
                expanded.append(line_clean)
        return expanded[:4] # Return original + up to 3 expansions
    except Exception:
        # Fallback to just the original query if LLM fails
        return [query]

class AdvancedRetriever:
    """Implements Hybrid Vector + BM25 search with Flashrank reranking."""
    
    def __init__(self, db_manager, llm):
        self.db_manager = db_manager
        self.llm = llm
        self.bm25 = None
        self.bm25_docs = []
        self.rebuild_bm25()

    def rebuild_bm25(self):
        """
        Pull all documents from the Chroma database to compile the BM25 index 
        for lexical keyword search.
        """
        if self.db_manager.db is None:
            self.bm25 = None
            self.bm25_docs = []
            return
            
        try:
            # Retrieve all records from Chroma collection
            data = self.db_manager.db.get()
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            
            if not documents:
                self.bm25 = None
                self.bm25_docs = []
                return
                
            self.bm25_docs = []
            tokenized_corpus = []
            
            for doc, meta in zip(documents, metadatas):
                langchain_doc = Document(page_content=doc, metadata=meta or {})
                self.bm25_docs.append(langchain_doc)
                # Simple tokenization by splitting and lowercasing
                tokenized_corpus.append(doc.lower().split())
                
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception:
            self.bm25 = None
            self.bm25_docs = []

    def retrieve(self, query: str, top_k: int = 5, run_expansion: bool = True) -> list[dict]:
        """
        Runs the full Advanced Retrieval pipeline:
        1. Query Expansion (optional)
        2. Vector Search (Chroma) + Lexical Search (BM25) -> Hybrid candidate list
        3. Flashrank Reranking -> Output top_k documents
        """
        # Expand query
        queries = expand_query(query, self.llm) if run_expansion else [query]
        
        candidates = {} # Map page_content hash to (Document, source_type_score)
        
        # 1. Vector Search Candidates
        vector_store = self.db_manager.db
        if vector_store:
            for q in queries:
                # Retrieve top 5 from vector search for each query variation
                res = vector_store.similarity_search_with_relevance_scores(q, k=5)
                for doc, score in res:
                    content = doc.page_content
                    if content not in candidates:
                        candidates[content] = {
                            "doc": doc,
                            "vector_score": float(score),
                            "bm25_score": 0.0,
                            "source": "vector"
                        }
                    else:
                        candidates[content]["vector_score"] = max(candidates[content]["vector_score"], float(score))

        # 2. BM25 Search Candidates
        # If BM25 is not built yet (or db changed), rebuild it
        if self.bm25 is None and vector_store:
            self.rebuild_bm25()
            
        if self.bm25:
            for q in queries:
                tokenized_query = q.lower().split()
                # Get BM25 raw scores
                scores = self.bm25.get_scores(tokenized_query)
                # Select top 5 BM25 index indices
                top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:5]
                
                for idx in top_indices:
                    score = scores[idx]
                    if score <= 0:
                        continue
                    doc = self.bm25_docs[idx]
                    content = doc.page_content
                    if content not in candidates:
                        candidates[content] = {
                            "doc": doc,
                            "vector_score": 0.0,
                            "bm25_score": float(score),
                            "source": "bm25"
                        }
                    else:
                        candidates[content]["bm25_score"] = max(candidates[content]["bm25_score"], float(score))
                        candidates[content]["source"] = "hybrid"

        candidate_list = list(candidates.values())
        if not candidate_list:
            return []
            
        # 3. Flashrank Reranking
        try:
            ranker = get_flashrank_ranker()
            
            # Translate candidates to flashrank format
            flash_passages = []
            for i, c in enumerate(candidate_list):
                flash_passages.append({
                    "id": i,
                    "text": c["doc"].page_content,
                    "meta": c["doc"].metadata
                })
                
            rerank_request = RerankRequest(query=query, passages=flash_passages)
            results = ranker.rerank(rerank_request)
            
            # Format results in descending order of score
            final_docs = []
            for r in results[:top_k]:
                idx = r["id"]
                original_candidate = candidate_list[idx]
                
                final_docs.append({
                    "document": original_candidate["doc"],
                    "rerank_score": float(r["score"]),
                    "vector_score": original_candidate["vector_score"],
                    "bm25_score": original_candidate["bm25_score"],
                    "source": original_candidate["source"]
                })
                
            return final_docs
            
        except Exception as e:
            # Fallback to combining score sorting if reranking fails
            # Sort by max score
            candidate_list.sort(key=lambda x: max(x["vector_score"], x["bm25_score"] / 20.0), reverse=True)
            
            final_docs = []
            for c in candidate_list[:top_k]:
                final_docs.append({
                    "document": c["doc"],
                    "rerank_score": 0.0, # Not computed
                    "vector_score": c["vector_score"],
                    "bm25_score": c["bm25_score"],
                    "source": c["source"]
                })
            return final_docs
