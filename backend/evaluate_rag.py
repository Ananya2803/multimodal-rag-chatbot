import os
import argparse
import pickle
import numpy as np
from vector_store import VectorStore

DB_PATH = "qdrant_db"

def print_database_summary(store):
    print("=" * 60)
    print(" RAG DATABASE SUMMARY")
    print("=" * 60)
    
    total = len(store.chunks)
    print(f"Total chunks in database: {total}")
    
    counts = {"text": 0, "table": 0, "image": 0}
    for c in store.chunks:
        ctype = c.get("type", "text")
        counts[ctype] = counts.get(ctype, 0) + 1
        
    for ctype, count in counts.items():
        print(f" - {ctype.capitalize()} chunks: {count} ({count/total*100:.1f}%)" if total > 0 else f" - {ctype.capitalize()} chunks: 0")
        
    print("\n--- SAMPLE CHUNKS ---")
    samples = {}
    for c in store.chunks:
        ctype = c.get("type", "text")
        if ctype not in samples:
            samples[ctype] = c
            
    for ctype, c in samples.items():
        print(f"\n[{ctype.upper()} SAMPLE] (Page {c['page']}, Source: {c['doc_name']})")
        print(c["content"][:300] + "\n...")
    print("=" * 60)

def search_and_evaluate(store, query, top_k=5, hf_token=None, show_neighbors=False):
    print("\n" + "=" * 60)
    print(f" RAG SEARCH EVALUATION FOR QUERY: '{query}'")
    print("=" * 60)
    
    # Run retrieval
    results = store.search(query, top_k=top_k, hf_token=hf_token)
    if not results:
        print("No matches found.")
        return
        
    for idx, r in enumerate(results):
        chunk = r["chunk"]
        doc = chunk["doc_name"]
        page = chunk["page"]
        ctype = chunk["type"]
        raw_score = r["raw_score"]
        boosted_score = r["boosted_score"]
        priority = r["priority"]
        
        # Find index in store.chunks to enable neighbor lookup
        chunk_idx = -1
        for i, c in enumerate(store.chunks):
            if c["content"] == chunk["content"] and c["doc_name"] == doc and c["page"] == page:
                chunk_idx = i
                break
                
        print(f"\nMatch #{idx+1} | Score: {raw_score:.4f} (Boosted: {boosted_score:.4f}, Priority: {priority})")
        print(f"Chunk Index: {chunk_idx} | Type: {ctype.upper()} | Page: {page} | Source: {doc}")
        print("-" * 50)
        print(chunk["content"][:500])
        if len(chunk["content"]) > 500:
            print("...")
            
        if show_neighbors and chunk_idx != -1:
            print("\n  [Neighboring Chunks]")
            if chunk_idx > 0:
                prev_c = store.chunks[chunk_idx - 1]
                if prev_c["doc_name"] == doc:
                    print(f"   <- PREV (Index {chunk_idx-1}, Page {prev_c['page']}, {prev_c['type'].upper()}): {prev_c['content'][:150].replace('\n', ' ')}...")
            if chunk_idx < len(store.chunks) - 1:
                next_c = store.chunks[chunk_idx + 1]
                if next_c["doc_name"] == doc:
                    print(f"   -> NEXT (Index {chunk_idx+1}, Page {next_c['page']}, {next_c['type'].upper()}): {next_c['content'][:150].replace('\n', ' ')}...")
        print("-" * 50)
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate and debug RAG vector store retrieval.")
    parser.add_argument("query", nargs="?", default=None, help="The search query or keyword.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--neighbors", action="store_true", help="Show neighboring chunks for results.")
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at {DB_PATH}. Please run rebuild_db.py first.")
        return
        
    # Load store
    store = VectorStore(storage_path=DB_PATH)
    
    # Load HF_TOKEN from environment if available
    ENV_PATH = ".env"
    hf_token = None
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() == "HF_TOKEN":
                        hf_token = val.strip().strip('"').strip("'")
                        
    if args.query is None:
        print_database_summary(store)
    else:
        search_and_evaluate(store, args.query, top_k=args.top_k, hf_token=hf_token, show_neighbors=args.neighbors)

if __name__ == "__main__":
    main()
