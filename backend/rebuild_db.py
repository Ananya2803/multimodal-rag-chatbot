import os
import logging
from pdf_parser import parse_pdf
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads"
IMAGE_DIR = "static/images"
DB_PATH = "qdrant_db"

# Load local .env file manually if it exists
ENV_PATH = ".env"
hf_token = None
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if key.strip() == "HF_TOKEN":
                        hf_token = val.strip().strip('"').strip("'")
        logger.info("Loaded HF_TOKEN from local .env file.")
    except Exception as e:
        logger.error(f"Error loading local .env file: {e}")

def rebuild_database():
    # Initialize vector store
    vector_store = VectorStore(storage_path=DB_PATH)
    
    # 1. Clear database
    logger.info("Clearing existing database...")
    vector_store.clear()
    
    # 2. Find PDFs in uploads
    if not os.path.exists(UPLOAD_DIR):
        logger.info("No uploads directory found.")
        return
        
    pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        logger.info("No PDFs found in uploads directory to rebuild.")
        return
        
    logger.info(f"Found {len(pdfs)} PDFs to re-process: {pdfs}")
    
    for pdf_name in pdfs:
        pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
        logger.info(f"Re-parsing and embedding PDF: {pdf_name}...")
        try:
            chunks = parse_pdf(pdf_path, output_image_dir=IMAGE_DIR, hf_token=hf_token)
            
            # Count chunk types
            text_count = sum(1 for c in chunks if c["type"] == "text")
            table_count = sum(1 for c in chunks if c["type"] == "table")
            image_count = sum(1 for c in chunks if c["type"] == "image")
            
            logger.info(f"Extracted {len(chunks)} total chunks (text: {text_count}, tables: {table_count}, images: {image_count})")
            
            if chunks:
                vector_store.add_documents(chunks, hf_token=hf_token)
                logger.info(f"Successfully added {pdf_name} chunks to database.")
            else:
                logger.warning(f"No chunks extracted from {pdf_name}.")
                
        except Exception as e:
            logger.error(f"Failed to process {pdf_name}: {e}")
            
    logger.info("Database rebuild completed successfully!")

if __name__ == "__main__":
    rebuild_database()
