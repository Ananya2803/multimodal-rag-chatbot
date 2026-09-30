import os
import fitz  # PyMuPDF
from pdf_parser import parse_pdf
from vector_store import VectorStore

def create_test_pdfs():
    """Generates two simple PDF files to test RAG and prioritization."""
    print("Generating test PDF files...")
    
    # 1. Document A: Official Guidelines (High priority source)
    doc_a = fitz.open()
    page = doc_a.new_page()
    page.insert_text(
        (50, 50), 
        "Official Company Guidelines 2026", 
        fontsize=16, 
        color=(0, 0, 1) # Blue
    )
    page.insert_text(
        (50, 100), 
        "Refund policy: Customers are eligible for a full refund within 30 days of purchase.",
        fontsize=11
    )
    
    # Insert Table representation in text
    table_text = (
        "Pricing Table:\n"
        "Plan | Price | Storage\n"
        "Basic | $10/mo | 100 GB\n"
        "Pro | $30/mo | 1 TB\n"
        "Enterprise | $100/mo | 10 TB"
    )
    page.insert_text((50, 150), table_text, fontsize=10)
    doc_a.save("test_official_doc.pdf")
    doc_a.close()
    
    # 2. Document B: Unofficial Blog Post (Low priority source)
    doc_b = fitz.open()
    page = doc_b.new_page()
    page.insert_text(
        (50, 50), 
        "My Personal Blog - Tech Thoughts", 
        fontsize=16, 
        color=(1, 0.5, 0) # Orange
    )
    page.insert_text(
        (50, 100), 
        "Refund policy: I heard from a friend that they sometimes give refunds up to 60 days.",
        fontsize=11
    )
    doc_b.save("test_blog_post.pdf")
    doc_b.close()
    
    print("Test PDFs created: test_official_doc.pdf, test_blog_post.pdf")

def run_tests():
    # Setup test PDFs
    create_test_pdfs()
    
    # Initialize vector store
    db_path = "test_qdrant_db"
    if os.path.exists(db_path):
        import shutil
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)
        
    store = VectorStore(storage_path=db_path)
    
    # Parse PDF A
    print("\n--- Parsing test_official_doc.pdf ---")
    chunks_a = parse_pdf("test_official_doc.pdf", output_image_dir="test_images")
    print(f"Extracted {len(chunks_a)} chunks.")
    for c in chunks_a:
        print(f"[{c['type']}] (Page {c['page']}): {c['content'][:150]}...")
        
    # Parse PDF B
    print("\n--- Parsing test_blog_post.pdf ---")
    chunks_b = parse_pdf("test_blog_post.pdf", output_image_dir="test_images")
    print(f"Extracted {len(chunks_b)} chunks.")
    for c in chunks_b:
        print(f"[{c['type']}] (Page {c['page']}): {c['content'][:150]}...")
        
    # Add to vector store
    print("\nEmbedding chunks and adding to VectorStore...")
    store.add_documents(chunks_a + chunks_b)
    
    # Test query 1: Default priorities (Equal weight, standard retrieval)
    query = "What is the refund policy?"
    print(f"\n--- Searching with Equal Weights (Standard Mode, Query: '{query}') ---")
    results = store.search(query, top_k=3, retrieval_mode="standard")
    for r in results:
        print(f"Doc: {r['chunk']['doc_name']} | Raw Score: {r['raw_score']:.4f} | Boosted: {r['boosted_score']:.4f} | Priority: {r['priority']}")
        print(f"Content: {r['chunk']['content']}\n")
        
    # Set priorities: Official Doc (High) and Blog Post (Low)
    print("\nSetting priorities: official doc -> High, blog post -> Low")
    store.set_priority("test_official_doc.pdf", "High")
    store.set_priority("test_blog_post.pdf", "Low")
    
    # Test query 2: Standard mode with priorities
    print(f"\n--- Searching with Weighted Priorities (Standard Mode, Query: '{query}') ---")
    results = store.search(query, top_k=3, retrieval_mode="standard")
    for r in results:
        print(f"Doc: {r['chunk']['doc_name']} | Raw Score: {r['raw_score']:.4f} | Boosted: {r['boosted_score']:.4f} | Priority: {r['priority']}")
        
    # Test query 3: Tiered Priority Fallback (Fact in both documents)
    print(f"\n--- Tiered Priority Fallback (Query: '{query}') ---")
    results = store.search(query, top_k=2, retrieval_mode="tiered", relevance_threshold=0.1)
    for r in results:
        print(f"Doc: {r['chunk']['doc_name']} | Raw Score: {r['raw_score']:.4f} | Priority: {r['priority']}")
        print(f"Content: {r['chunk']['content']}\n")
    # Verify High priority document is returned first
    if results and results[0]['chunk']['doc_name'] == "test_official_doc.pdf":
        print("✅ SUCCESS: High priority document chunk retrieved first.")
    else:
        print("❌ FAILURE: High priority document chunk was not retrieved first.")

    # Test query 4: Tiered Priority Fallback (Fact ONLY in Low-priority document)
    low_only_query = "Tech Thoughts Blog Post"
    print(f"\n--- Tiered Priority Fallback (Fact only in Low Priority Doc, Query: '{low_only_query}') ---")
    results = store.search(low_only_query, top_k=2, retrieval_mode="tiered", relevance_threshold=0.1)
    for r in results:
        print(f"Doc: {r['chunk']['doc_name']} | Raw Score: {r['raw_score']:.4f} | Priority: {r['priority']}")
        print(f"Content: {r['chunk']['content']}\n")
    # Verify Low priority document is retrieved since no High priority document matches it
    if results and any(r['chunk']['doc_name'] == "test_blog_post.pdf" for r in results):
        print("✅ SUCCESS: Successfully fell back to Low-priority document when fact was only there.")
    else:
        print("❌ FAILURE: Failed to retrieve Low-priority document chunk.")

    # Clean up test files
    for filename in ["test_official_doc.pdf", "test_blog_post.pdf", db_path]:
        if os.path.exists(filename):
            if os.path.isdir(filename):
                import shutil
                shutil.rmtree(filename)
            else:
                os.remove(filename)
            
    print("Tests completed and cleaned up.")

if __name__ == "__main__":
    run_tests()
