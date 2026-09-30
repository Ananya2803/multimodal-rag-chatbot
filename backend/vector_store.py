import os
import json
import numpy as np
from huggingface_hub import InferenceClient
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

# Map human-readable priority levels to multiplier weights
PRIORITY_WEIGHTS = {
    "High": 1.5,
    "Medium": 1.0,
    "Low": 0.5
}

class VectorStore:
    def __init__(self, storage_path="qdrant_db"):
        self.qdrant_path = storage_path
        self.collection_name = "chunks"
        self.priorities_path = os.path.join(self.qdrant_path, "priorities.json")
        
        # Connect to local Qdrant persisted DB
        self.client = QdrantClient(path=self.qdrant_path)
        self._ensure_collections()
        
        self.chunks = []       # List of chunk dicts
        self.embeddings = []   # List of list/array of embeddings
        self.priorities = {}   # Dict of doc_name -> priority string ("High", "Medium", "Low")
        
        self.load()

    def _ensure_collections(self):
        """Creates the Qdrant chunks collection if it does not already exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")

    def save_priorities(self):
        """Saves priorities locally in a JSON file."""
        try:
            with open(self.priorities_path, "w") as f:
                json.dump(self.priorities, f, indent=2)
            logger.info(f"Priorities saved successfully to {self.priorities_path}.")
        except Exception as e:
            logger.error(f"Error saving priorities: {e}")

    def load_priorities(self):
        """Loads priorities from local JSON file."""
        if os.path.exists(self.priorities_path):
            try:
                with open(self.priorities_path, "r") as f:
                    self.priorities = json.load(f)
                logger.info(f"Loaded priorities for {len(self.priorities)} documents.")
            except Exception as e:
                logger.error(f"Error loading priorities: {e}")
                self.priorities = {}
        else:
            self.priorities = {}

    def save(self):
        """No-op for backward compatibility. Qdrant auto-commits to disk."""
        self.save_priorities()

    def load(self):
        """Loads all vectors and chunks from Qdrant into memory to keep backward compatibility."""
        self.chunks = []
        self.embeddings = []
        self.load_priorities()
        
        try:
            if self.client.collection_exists(self.collection_name):
                offset = None
                while True:
                    response = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        with_payload=True,
                        with_vectors=True,
                        offset=offset
                    )
                    points, offset = response
                    for p in points:
                        chunk = p.payload
                        self.chunks.append(chunk)
                        # Qdrant returns vectors as a list, convert to numpy array
                        self.embeddings.append(np.array(p.vector))
                        
                    if offset is None:
                        break
                logger.info(f"Vector store loaded from Qdrant. Loaded {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error loading vector store from Qdrant: {e}")
            self.chunks = []
            self.embeddings = []

    def get_embeddings_hf(self, texts, hf_token=None, model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Fetches embeddings from Hugging Face Inference API.
        Can process a single string or a list of strings.
        Raises an exception if the API call fails or if no token is provided.
        """
        if not texts:
            return []
            
        if not hf_token:
            raise ValueError("No Hugging Face token provided")
            
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        client = InferenceClient(token=hf_token)
        embeddings = client.feature_extraction(text_list, model=model)
        
        # Convert embeddings to numpy array if it is not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        # Ensure it has 2 dimensions (batch size, embedding dimension)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
            
        if embeddings.size > 0:
            return embeddings
            
        raise RuntimeError("InferenceClient feature_extraction returned empty or unexpected result.")

    def add_documents(self, chunks, hf_token=None):
        """
        Embeds a list of chunks and adds them to Qdrant and in-memory caches.
        """
        if not chunks:
            return
            
        # Extract text content from chunks to embed
        texts = [c["content"] for c in chunks]
        
        # Hugging Face Inference client has a rate limit or payload limit, so let's batch requests
        batch_size = 32
        new_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embs = self.get_embeddings_hf(batch_texts, hf_token=hf_token)
                new_embeddings.extend(batch_embs)
            except Exception as e:
                logger.warning(f"Error generating embeddings: {e}. Storing fallback zero embeddings for BM25 keyword search compatibility.")
                new_embeddings.extend(np.zeros((len(batch_texts), 384)))
            
        # Initialize default priority for new documents
        for chunk in chunks:
            doc_name = chunk["doc_name"]
            if doc_name not in self.priorities:
                self.priorities[doc_name] = "Medium"
                
        # Prep points for Qdrant
        points = []
        import uuid
        import hashlib
        for chunk, emb in zip(chunks, new_embeddings):
            # Deterministic UUID based on content to prevent duplicate points
            h = hashlib.md5(f"{chunk['doc_name']}_{chunk['page']}_{chunk['type']}_{chunk['content']}".encode("utf-8")).hexdigest()
            point_id = str(uuid.UUID(h))
            
            points.append(PointStruct(
                id=point_id,
                vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                payload=chunk
            ))
            
        # Upsert in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points into Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to upsert points into Qdrant: {e}")
            
        # Save priorities
        self.save_priorities()
        
        # Update in-memory lists to keep backward compatibility
        self.chunks.extend(chunks)
        self.embeddings.extend(new_embeddings)

    def set_priority(self, doc_name, priority):
        """Sets the priority of a document and saves it."""
        if priority in PRIORITY_WEIGHTS:
            self.priorities[doc_name] = priority
            self.save_priorities()
            logger.info(f"Priority for {doc_name} set to {priority}.")
        else:
            logger.error(f"Invalid priority level: {priority}")

    def clear(self):
        """Clears all vectors and documents."""
        self.chunks = []
        self.embeddings = []
        self.priorities = {}
        
        # Clear Qdrant collection
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info("Qdrant collection cleared.")
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {e}")
            
        # Remove local priorities file
        if os.path.exists(self.priorities_path):
            try:
                os.remove(self.priorities_path)
            except Exception as e:
                logger.error(f"Error deleting priorities file: {e}")
                
        logger.info("Vector store cleared.")

    def search(self, query, top_k=5, hf_token=None, retrieval_mode="tiered", relevance_threshold=0.15):
        """
        Searches the vector store for a query, using hybrid search (semantic + BM25)
        and sorting based on the specified retrieval mode:
        - 'tiered' (default): Priority fallback retrieval based on source reliability.
        - 'standard': Pure similarity-based ranking with priority weights.
        - 'diverse': Round-robin cross-document diverse selection.
        """
        if not self.chunks:
            return []
            
        # 1. Compute BM25 scores
        import re
        from collections import Counter
        import math
        
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
            
        query_tokens = tokenize(query)
        
        # Compute Document Frequency (DF) for each query term
        df = {}
        for token in set(query_tokens):
            df[token] = sum(1 for c in self.chunks if token in tokenize(c["content"]))
            
        # Compute IDF
        idf = {}
        num_chunks = len(self.chunks)
        for token, count in df.items():
            idf[token] = math.log((num_chunks - count + 0.5) / (count + 0.5) + 1.0)
            if idf[token] < 0.0001:
                idf[token] = 0.0001
                
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        
        chunk_lengths = [len(tokenize(c["content"])) for c in self.chunks]
        avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 1.0
        
        bm25_dict = {}
        bm25_scores = []
        for idx, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk["content"])
            counter = Counter(tokens)
            doc_len = len(tokens)
            
            score = 0.0
            for token in query_tokens:
                if token in counter:
                    tf = counter[token]
                    num = tf * (k1 + 1)
                    den = tf + k1 * (1 - b + b * (doc_len / avg_len))
                    score += idf.get(token, 0) * (num / den)
            bm25_scores.append(score)
            
            key = (chunk["doc_name"], chunk["page"], chunk["type"], chunk["content"])
            bm25_dict[key] = score
            
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        
        # 2. Compute Semantic scores using Qdrant Index
        use_semantic = False
        query_emb = None
        
        # Check if DB has valid (non-zero) embeddings
        db_has_embeddings = False
        if len(self.embeddings) > 0:
            db_embeddings = np.array(self.embeddings)
            if not np.all(db_embeddings == 0):
                db_has_embeddings = True
                
        if hf_token and db_has_embeddings:
            try:
                query_emb = self.get_embeddings_hf(query, hf_token=hf_token)[0]
                use_semantic = True
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}. Running BM25 only.")
                
        qdrant_semantic_dict = {}
        if use_semantic and query_emb is not None:
            try:
                query_vector = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
                # Search up to 200 candidates for hybrid score blending
                search_limit = min(200, len(self.chunks))
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=search_limit,
                    with_payload=True
                )
                for r in search_results:
                    chunk = r.payload
                    key = (chunk["doc_name"], chunk["page"], chunk["type"], chunk["content"])
                    qdrant_semantic_dict[key] = float(r.score)
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}. Falling back to in-memory cosine similarity.")
                # Fallback to local cosine similarity
                query_emb = np.array(query_emb)
                db_embeddings = np.array(self.embeddings)
                dot_products = np.dot(db_embeddings, query_emb)
                db_norms = np.linalg.norm(db_embeddings, axis=1)
                query_norm = np.linalg.norm(query_emb)
                db_norms[db_norms == 0] = 1e-10
                if query_norm == 0:
                    query_norm = 1e-10
                cosine_similarities = dot_products / (db_norms * query_norm)
                for idx, chunk in enumerate(self.chunks):
                    key = (chunk["doc_name"], chunk["page"], chunk["type"], chunk["content"])
                    qdrant_semantic_dict[key] = float(cosine_similarities[idx])
            
        # 3. Combine scores
        results = []
        for chunk in self.chunks:
            doc_name = chunk["doc_name"]
            priority = self.priorities.get(doc_name, "Medium")
            
            key = (chunk["doc_name"], chunk["page"], chunk["type"], chunk["content"])
            
            # BM25 score
            raw_bm25 = bm25_dict.get(key, 0.0)
            norm_bm25 = (raw_bm25 / max_bm25) * 0.95 if max_bm25 > 0.0 else 0.0
            
            # Semantic score (default to 0.0 if not returned in Qdrant top 200 candidates)
            raw_semantic = qdrant_semantic_dict.get(key, 0.0)
            
            # Hybrid score blending (60% Semantic + 40% BM25)
            if use_semantic:
                raw_score = 0.6 * raw_semantic + 0.4 * norm_bm25
            else:
                raw_score = norm_bm25
                
            # Apply priority weight as a balanced additive boost (e.g. +0.1 for High, -0.1 for Low)
            if priority == "High":
                boost = 0.1
            elif priority == "Low":
                boost = -0.1
            else:
                boost = 0.0
            boosted_score = raw_score + boost
            
            results.append({
                "chunk": chunk,
                "raw_score": raw_score,
                "boosted_score": boosted_score,
                "priority": priority,
                "weight": PRIORITY_WEIGHTS.get(priority, 1.0)
            })
            
        if retrieval_mode == "tiered":
            # Group relevant chunks by priority tier
            high_tier = []
            medium_tier = []
            low_tier = []
            
            for r in results:
                # Filter by relevance threshold
                if r["raw_score"] >= relevance_threshold:
                    p = r["priority"]
                    if p == "High":
                        high_tier.append(r)
                    elif p == "Low":
                        low_tier.append(r)
                    else:
                        medium_tier.append(r)
                        
            # Sort each tier by similarity score descending (raw similarity is fine as weights are identical within a tier)
            high_tier.sort(key=lambda x: x["raw_score"], reverse=True)
            medium_tier.sort(key=lambda x: x["raw_score"], reverse=True)
            low_tier.sort(key=lambda x: x["raw_score"], reverse=True)
            
            # Check if High tier has strong matches.
            # If the best High tier match is weak (raw_score < 0.4) and Medium/Low has a much stronger match,
            # we should include Medium/Low matches.
            high_is_strong = len(high_tier) > 0 and high_tier[0]["raw_score"] >= 0.4
            
            selected_results = []
            if high_is_strong:
                # High priority info is present. Fill up to top_k with High priority,
                # but also allow exceptionally strong matches from Medium/Low tier (e.g. raw_score >= 0.6)
                # to be included so we don't miss them if the user specifically asked for them.
                selected_results.extend(high_tier[:top_k])
                
                # Check for exceptionally strong Medium/Low matches to insert
                strong_other = [r for r in (medium_tier + low_tier) if r["raw_score"] >= 0.6]
                strong_other.sort(key=lambda x: x["raw_score"], reverse=True)
                
                for r in strong_other:
                    if r not in selected_results:
                        if len(selected_results) < top_k:
                            selected_results.append(r)
                        else:
                            # Replace the weakest High tier chunk if the Medium/Low chunk is significantly better
                            selected_results.sort(key=lambda x: x["raw_score"])
                            if r["raw_score"] - selected_results[0]["raw_score"] >= 0.05:
                                selected_results[0] = r
            else:
                # High priority info is absent or weak. Fall back to Medium and Low.
                # Sort all tiers together by boosted score to get the best overall matches.
                combined = high_tier + medium_tier + low_tier
                combined.sort(key=lambda x: x["boosted_score"], reverse=True)
                selected_results = combined[:top_k]
                
            # If no matches passed the threshold, check if the best overall match is somewhat relevant
            if not selected_results:
                results.sort(key=lambda x: x["boosted_score"], reverse=True)
                best_raw_score = results[0]["raw_score"] if results else -1.0
                if best_raw_score >= 0.20:
                    logger.info(f"No chunks passed relevance threshold {relevance_threshold}, but best match has raw score {best_raw_score:.3f}. Falling back.")
                    return results[:min(top_k, 3)]
                else:
                    logger.info(f"All matches are irrelevant (best raw score: {best_raw_score:.3f}). Returning empty results.")
                    return []
                
            # Re-sort final selection by boosted score descending
            selected_results.sort(key=lambda x: x["boosted_score"], reverse=True)
            return selected_results
            
        elif retrieval_mode == "diverse":
            # Round-robin selection among active documents
            doc_groups = {}
            for r in results:
                if r["raw_score"] >= relevance_threshold:
                    doc_name = r["chunk"]["doc_name"]
                    if doc_name not in doc_groups:
                        doc_groups[doc_name] = []
                    doc_groups[doc_name].append(r)
                    
            # Sort chunks within each document group by boosted score
            for doc_name in doc_groups:
                doc_groups[doc_name].sort(key=lambda x: x["boosted_score"], reverse=True)
                
            selected_results = []
            docs = list(doc_groups.keys())
            if docs:
                idx_ptr = {doc: 0 for doc in docs}
                has_items = True
                while has_items and len(selected_results) < top_k:
                    has_items = False
                    for doc in docs:
                        ptr = idx_ptr[doc]
                        if ptr < len(doc_groups[doc]):
                            selected_results.append(doc_groups[doc][ptr])
                            idx_ptr[doc] += 1
                            has_items = True
                            if len(selected_results) >= top_k:
                                break
                                
            # If no matches passed the threshold, check if the best overall match is somewhat relevant
            if not selected_results:
                results.sort(key=lambda x: x["boosted_score"], reverse=True)
                best_raw_score = results[0]["raw_score"] if results else -1.0
                if best_raw_score >= 0.20:
                    logger.info(f"No chunks passed relevance threshold {relevance_threshold}, but best match has raw score {best_raw_score:.3f}. Falling back.")
                    return results[:min(top_k, 3)]
                else:
                    logger.info(f"All matches are irrelevant (best raw score: {best_raw_score:.3f}). Returning empty results.")
                    return []
                
            return selected_results
            
        else:
            # Standard weight-boosted retrieval
            results.sort(key=lambda x: x["boosted_score"], reverse=True)
            best_raw_score = results[0]["raw_score"] if results else -1.0
            if best_raw_score >= 0.20:
                return results[:top_k]
            else:
                logger.info(f"Standard Search: best raw score {best_raw_score:.3f} is too low. Returning empty results.")
                return []
