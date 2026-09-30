import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from huggingface_hub import InferenceClient

from pdf_parser import parse_pdf
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load local .env file manually if it exists
ENV_PATH = ".env"
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
        logger.info("Loaded environment variables from local .env file.")
    except Exception as e:
        logger.error(f"Error loading local .env file: {e}")

HF_TOKEN = os.getenv("HF_TOKEN")

from contextlib import asynccontextmanager

# Constants
UPLOAD_DIR = "uploads"
IMAGE_DIR = "static/images"
DB_PATH = "qdrant_db"

# Global placeholder for vector store
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    logger.info("Initializing VectorStore in lifespan startup...")
    vector_store = VectorStore(storage_path=DB_PATH)
    yield
    logger.info("Lifespan shutdown complete.")

app = FastAPI(title="RAG Chatbot Backend", lifespan=lifespan)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, configure to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mount static files to serve extracted images
app.mount("/static", StaticFiles(directory="static"), name="static")

class PriorityRequest(BaseModel):
    doc_name: str
    priority: str

class QueryRequest(BaseModel):
    query: str
    hf_token: Optional[str] = None
    model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct"
    top_k: Optional[int] = 5
    retrieval_mode: Optional[str] = "tiered"
    answer_style: Optional[str] = "detailed"

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    hf_token: Optional[str] = Form(None)
):
    """
    Uploads and parses multiple PDF files. Extracts text, tables, and images,
    adds them to the vector store.
    """
    parsed_stats = []
    token = hf_token if hf_token else HF_TOKEN
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
            
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save file to uploads folder
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Error saving uploaded file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save file {file.filename}")
            
        # Parse PDF
        try:
            logger.info(f"Parsing PDF: {file.filename}")
            chunks = parse_pdf(file_path, output_image_dir=IMAGE_DIR, hf_token=token)
            
            # Count chunk types
            text_count = sum(1 for c in chunks if c["type"] == "text")
            table_count = sum(1 for c in chunks if c["type"] == "table")
            image_count = sum(1 for c in chunks if c["type"] == "image")
            
            # Index chunks
            if chunks:
                vector_store.add_documents(chunks, hf_token=token)
                
            parsed_stats.append({
                "filename": file.filename,
                "text_chunks": text_count,
                "tables": table_count,
                "images": image_count,
                "total_chunks": len(chunks)
            })
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed parsing file {file.filename}")
            
    return {"status": "success", "uploads": parsed_stats}

@app.get("/api/documents")
async def get_documents():
    """
    Returns lists of uploaded documents with stats and current priority configurations.
    """
    docs = {}
    for chunk in vector_store.chunks:
        doc_name = chunk["doc_name"]
        if doc_name not in docs:
            file_path = os.path.join(UPLOAD_DIR, doc_name)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            docs[doc_name] = {
                "doc_name": doc_name,
                "size_bytes": file_size,
                "priority": vector_store.priorities.get(doc_name, "Medium"),
                "text_chunks": 0,
                "tables": 0,
                "images": 0,
                "pages": set()
            }
            
        docs[doc_name]["pages"].add(chunk["page"])
        if chunk["type"] == "text":
            docs[doc_name]["text_chunks"] += 1
        elif chunk["type"] == "table":
            docs[doc_name]["tables"] += 1
        elif chunk["type"] == "image":
            docs[doc_name]["images"] += 1
            
    # Convert pages set to count/list
    result = []
    for doc_name, data in docs.items():
        data["page_count"] = len(data["pages"])
        del data["pages"]
        result.append(data)
        
    return result

@app.post("/api/documents/priority")
async def set_document_priority(req: PriorityRequest):
    """
    Updates the priority of a document.
    """
    if req.doc_name not in vector_store.priorities:
         raise HTTPException(status_code=404, detail="Document not found")
         
    vector_store.set_priority(req.doc_name, req.priority)
    return {"status": "success", "doc_name": req.doc_name, "priority": req.priority}

@app.post("/api/documents/clear")
async def clear_documents():
    """
    Clears all documents, vector store index, and deletes local uploads/extracted images.
    """
    vector_store.clear()
    
    # Clean folders
    try:
        for file in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, file))
        for file in os.listdir(IMAGE_DIR):
            os.remove(os.path.join(IMAGE_DIR, file))
    except Exception as e:
        logger.error(f"Error cleaning folders: {e}")
        
    return {"status": "success", "message": "All database and files cleared."}

# Conversational / Greeting keywords to bypass RAG search and prevent irrelevant citation badges
GREETINGS_AND_CONVERSATIONAL = {
    "hello", "hi", "hey", "hola", "greetings", "howdy", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thank you very much", "appreciate it", "great", "awesome", "thank", "thx",
    "bye", "goodbye", "see you", "how are you", "who are you", "what are you", "help", "what can you do",
    "hi there", "hello there", "hey there", "good morning!", "good afternoon!", "good evening!",
    "nice to meet you", "how's it going", "how is it going", "what's up", "sup", "yo", "fine", "ok", "okay",
    "thank you!", "thanks!", "great!", "awesome!"
}

def is_conversational_query(query: str) -> bool:
    # Strip common leading/trailing punctuation and symbols
    clean = query.strip().lower().strip("?.!,;:*~`\"' ")
    if clean in GREETINGS_AND_CONVERSATIONAL:
        return True
    
    # If the query is very short (up to 3 words) and contains only common chat/greeting terms
    words = clean.split()
    if len(words) <= 3:
        chat_words = {
            "hello", "hi", "hey", "hola", "greetings", "thanks", "thank", "you", "bye", "goodbye", 
            "good", "morning", "afternoon", "evening", "howdy", "sup", "yo", "fine", "ok", "okay", 
            "great", "there", "assistant", "bot", "chatbot", "helper", "test", "testing", "help",
            "me", "please", "dear", "everyone", "team", "all", "now", "again", "today", "guy", "guys"
        }
        if all(w in chat_words for w in words):
            return True
    return False

import re

def filter_active_citations(answer: str, citations: list) -> list:
    """
    Filters the retrieved citations to only return those that are actually
    referenced by the LLM, or keeps the top matches if the LLM did not write brackets
    but answered the question successfully using the provided context.
    """
    if not citations or not answer:
        return []
        
    answer_lower = answer.lower()
    
    # 1. If the LLM indicates a negative fallback response, discard all citations
    negatives = [
        "could not find", "not found in the", "not mentioned in", 
        "no information", "does not contain", "no mention of", 
        "cannot find", "unable to find"
    ]
    if any(neg in answer_lower for neg in negatives):
        if "could not find" in answer_lower or "not mentioned" in answer_lower or "does not contain" in answer_lower:
            return []
            
    # Find all substrings inside square brackets [...] in the response text
    bracket_contents = re.findall(r'\[([^\]]+)\]', answer)
    bracket_contents_lower = [content.lower() for content in bracket_contents]
    
    active_citations = []
    
    # 2. If bracketed citations exist in the response, we strictly match them
    if bracket_contents_lower:
        for citation in citations:
            doc_name = citation["doc_name"]
            page = citation["page"]
            
            doc_lower = doc_name.lower().strip()
            doc_base = os.path.splitext(doc_lower)[0] if "." in doc_lower else doc_lower
            
            matched = False
            for content in bracket_contents_lower:
                if (doc_lower in content or doc_base in content) and str(page) in content:
                    matched = True
                    break
                    
            if not matched:
                # Fallback to check if mentioned in the nearby text without brackets
                if (doc_lower in answer_lower or doc_base in answer_lower):
                    if f"page {page}" in answer_lower or f"p. {page}" in answer_lower or f"p.{page}" in answer_lower:
                        matched = True
                        
            if matched:
                active_citations.append(citation)
    else:
        # 3. If the LLM did not output any bracketed citations at all (e.g. forgot formatting),
        # keep chunks that are highly relevant (raw_score >= 0.35) or are in the top 3.
        for citation in citations:
            score = citation.get("raw_score", 0.0)
            if score >= 0.35 or citation["index"] <= 3:
                active_citations.append(citation)
                
    # Re-index remaining citations to keep them continuous in the UI
    for idx, citation in enumerate(active_citations):
        citation["index"] = idx + 1
        
    return active_citations

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    """
    Searches matching context from RAG, formats prompt, queries HF LLM,
    and returns answer + citations.
    """
    token = req.hf_token if req.hf_token else HF_TOKEN
    
    # If it is a greeting or simple conversational chat, skip RAG and respond directly
    if is_conversational_query(req.query):
        logger.info("Conversational query detected. Responding directly without RAG context.")
        conversational_system_prompt = (
            "You are a helpful, intelligent RAG chatbot assistant. "
            "The user is greeting you or making general conversation. "
            "Please respond to their query politely and naturally as a friendly assistant."
        )
        try:
            client = InferenceClient(token=token)
            messages = [
                {"role": "system", "content": conversational_system_prompt},
                {"role": "user", "content": req.query}
            ]
            model_name = req.model if req.model else "Qwen/Qwen2.5-7B-Instruct"
            response = client.chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=256,
                temperature=0.7
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "citations": []
            }
        except Exception as e:
            logger.error(f"Error in conversational fallback LLM query: {e}")
            clean_q = req.query.strip().lower()
            if clean_q in {"hello", "hi", "hey", "hola", "greetings", "howdy"}:
                return {"answer": "Hello! How can I help you today? Please feel free to ask questions about your documents.", "citations": []}
            elif clean_q in {"thanks", "thank you", "thank you very much"}:
                return {"answer": "You're very welcome! Let me know if you need anything else.", "citations": []}
            return {"answer": "Hello! How can I help you today?", "citations": []}

    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload a PDF first.")
    
    # Search RAG
    top_k = req.top_k if req.top_k is not None else 5
    retrieval_mode = req.retrieval_mode if req.retrieval_mode else "tiered"
    
    results = vector_store.search(
        req.query, 
        top_k=top_k, 
        hf_token=token, 
        retrieval_mode=retrieval_mode
    )
    # If no results were found, fall back to a direct LLM response without RAG context
    if not results:
        logger.info("No relevant chunks retrieved. Responding to query directly using LLM conversational fallback.")
        conversational_system_prompt = (
            "You are a helpful, intelligent RAG chatbot assistant. "
            "The user's query could not be matched with any context in the uploaded documents. "
            "Please respond to their query politely and naturally as a general AI assistant. "
            "If they are greeting you, greet them back warmly. "
            "If they are asking a specific question, answer it to the best of your general knowledge "
            "but note that the answer was not found in their uploaded documents."
        )
        
        try:
            client = InferenceClient(token=token)
            messages = [
                {"role": "system", "content": conversational_system_prompt},
                {"role": "user", "content": req.query}
            ]
            model_name = req.model if req.model else "Qwen/Qwen2.5-7B-Instruct"
            
            response = client.chat_completion(
                messages=messages,
                model=model_name,
                max_tokens=512,
                temperature=0.7
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "citations": []
            }
        except Exception as e:
            logger.error(f"Error in conversational fallback LLM query: {e}")
            # If the LLM API fails, return a polite static response
            clean_q = req.query.strip().lower()
            if clean_q in {"hello", "hi", "hey", "hola", "greetings", "howdy"}:
                return {"answer": "Hello! How can I help you today? Please feel free to ask questions about your documents.", "citations": []}
            elif clean_q in {"thanks", "thank you", "thank you very much"}:
                return {"answer": "You're very welcome! Let me know if you need anything else.", "citations": []}
            return {
                "answer": "I couldn't find any relevant information in your uploaded documents to answer this question.",
                "citations": []
            }
         
    # Format context and citations
    context_blocks = []
    citations = []
    
    for idx, r in enumerate(results):
        chunk = r["chunk"]
        doc = chunk["doc_name"]
        page = chunk["page"]
        ctype = chunk["type"]
        content = chunk["content"]
        
        # Format for context block
        if ctype == "image":
            image_path = chunk["metadata"].get("image_path")
            block = f"[Source: {doc}, Page: {page}, Type: {ctype}, Image Path: {image_path}]\n{content}\n"
        else:
            block = f"[Source: {doc}, Page: {page}, Type: {ctype}]\n{content}\n"
        context_blocks.append(block)
        
        # Format citation for frontend representation
        citation = {
            "index": idx + 1,
            "doc_name": doc,
            "page": page,
            "type": ctype,
            "content": content,
            "raw_score": r["raw_score"],
            "boosted_score": r["boosted_score"],
            "priority": r["priority"]
        }
        if ctype == "image":
            citation["image_path"] = chunk["metadata"].get("image_path")
        citations.append(citation)
        
    context_str = "\n---\n".join(context_blocks)
    
    # Prompt formulation based on answer style
    answer_style = req.answer_style if req.answer_style else "detailed"
    
    if answer_style == "detailed":
        style_instructions = (
            "Provide a highly detailed, comprehensive, and logically sound response. "
            "Synthesize relevant points across different documents if applicable, break down complex concepts, "
            "and explain your reasoning step-by-step. Use headings, sections, bullet points, and tables to structure your "
            "explanation clearly. Always cite facts and be thorough in your analysis."
        )
    elif answer_style == "concise":
        style_instructions = (
            "Provide a brief, direct, and concise response. Answer the question immediately with minimal fluff "
            "while maintaining accuracy and referencing sources. Avoid long explanations."
        )
    else: # balanced
        style_instructions = (
            "Provide a balanced, structured, and clear response. Provide a reasonable level of detail to "
            "answer the question fully but avoid unnecessary verbosity. Use bullet points or short paragraphs where appropriate."
        )
        
    system_prompt = (
        "You are an intelligent, helpful RAG chatbot assistant.\n\n"
        f"Style Guideline:\n{style_instructions}\n\n"
        "Rule: Your task is to answer the user's question using ONLY the provided context below. "
        "Ensure you base your answers on the provided context. If the context does not contain "
        "the answer, explain that you couldn't find the information in the documents. "
        "Do not speculate or make things up.\n\n"
        "Citations guideline:\n"
        "Whenever you state a fact derived from a source, ALWAYS add a citation tag in the format [DocName, Page X] "
        "at the end of the sentence. (Example: 'The company's revenue increased by 15% in 2023 [annual_report.pdf, Page 4].').\n\n"
        "Format tables neatly using standard Markdown tables when displaying tabular data.\n\n"
        "Image rendering guideline:\n"
        "If a retrieved source chunk has Type: image, and the user's query refers to a map, graph, chart, photo, or image (or visual details on a page), "
        "you MUST render it in your response using standard Markdown image syntax: `![Image Description]({image_path})` "
        "(where {image_path} is the exact relative Image Path provided in the source header, e.g. /static/images/img_name.png). "
        "Never output raw image path strings as plain text; always wrap them in standard Markdown image tags `![...]({image_path})`. "
        "Explain what the image/map/chart contains in your response."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {req.query}"
    
    # Call HF API LLM
    try:
        client = InferenceClient(token=token)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use user-selected model or default Llama-3-8B-Instruct
        model_name = req.model if req.model else "meta-llama/Meta-Llama-3-8B-Instruct"
        logger.info(f"Querying HF model: {model_name}")
        
        response = client.chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=1024,
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        filtered_citations = filter_active_citations(answer, citations)
        return {
            "answer": answer,
            "citations": filtered_citations
        }
        
    except Exception as e:
        logger.error(f"Error calling HF LLM: {e}")
        # Return fallback response with citations so user still gets context
        error_msg = (
            f"An error occurred while connecting to the Hugging Face LLM API ({str(e)}).\n\n"
            "Here is the relevant context retrieved from your documents:\n\n"
        )
        for c in citations:
            error_msg += f"- **[{c['doc_name']}, Page {c['page']}]** (Score: {c['boosted_score']:.2f}):\n"
            error_msg += f"  {c['content'][:300]}...\n\n"
            
        return {
            "answer": error_msg,
            "citations": citations,
            "error": str(e)
        }
