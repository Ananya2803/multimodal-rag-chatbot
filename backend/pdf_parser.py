import os
import io
import fitz  # PyMuPDF
import pdfplumber
import subprocess
import httpx
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def format_table_as_markdown(table):
    """
    Formats a table (list of lists of cells) into a Markdown table.
    """
    if not table or not any(table):
        return ""
    
    # Ensure all rows have the same width
    max_cols = max(len(row) for row in table if row)
    cleaned_table = []
    
    for row in table:
        if not row:
            continue
        cleaned_row = []
        for cell in row:
            # Clean newlines and double pipes
            val = str(cell).replace("\n", " ").replace("|", "\\|").strip() if cell is not None else ""
            cleaned_row.append(val)
        # Pad row if short
        if len(cleaned_row) < max_cols:
            cleaned_row += [""] * (max_cols - len(cleaned_row))
        cleaned_table.append(cleaned_row)
        
    if not cleaned_table:
        return ""
        
    headers = cleaned_table[0]
    # Handle case where headers are completely empty
    if not any(headers):
        headers = [f"Col {i+1}" for i in range(len(headers))]
        
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for row in cleaned_table[1:]:
        md_table += "| " + " | ".join(row) + " |\n"
        
    return md_table

import re

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits text into overlapping chunks along sentence boundaries.
    Never splits words in half.
    """
    if not text:
        return []
        
    # Split text into sentences (lookbehind for sentence terminators followed by whitespace)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        # If a single sentence exceeds chunk_size, split it cleanly by words
        if sentence_len > chunk_size:
            # Output any accumulated chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            words = sentence.split()
            word_chunk = []
            word_len = 0
            for word in words:
                if word_len + len(word) + 1 > chunk_size:
                    if word_chunk:
                        chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                    word_len = len(word)
                else:
                    word_chunk.append(word)
                    word_len += len(word) + 1
            if word_chunk:
                # Seed the next iteration with the rest of the split sentence
                current_chunk = word_chunk
                current_length = word_len
            continue
            
        if current_length + sentence_len + 1 > chunk_size:
            # Save the current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Extract sentences to form the overlap for the next chunk
            overlap_chunk = []
            overlap_len = 0
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) + 1 <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_len += len(prev_sent) + 1
                else:
                    break
            
            current_chunk = overlap_chunk + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_length > 1 else 0)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

HF_API_FAILED = False
CONSECUTIVE_FAILURES = 0

def describe_image(image_bytes, hf_token=None):
    """
    Calls Hugging Face Inference API directly via httpx POST to generate a caption for the image.
    Bypasses high-level InferenceClient wrappers to avoid python 3.14 library compatibility issues.
    If it fails, returns a generic but valid visual asset description.
    """
    global HF_API_FAILED, CONSECUTIVE_FAILURES
    default_desc = "Visual chart, diagram, map, or illustration from the document."
    
    if HF_API_FAILED:
        return default_desc
        
    if not hf_token:
        return default_desc
        
    # Try a few models in case one is offline or not supported
    models = [
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
        "nlpconnect/vit-gpt2-image-captioning"
    ]
    
    for model in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            # Set timeout to 2 seconds to fail fast
            res = httpx.post(url, headers=headers, content=image_bytes, timeout=2.0)
            if res.status_code == 200:
                res_json = res.json()
                CONSECUTIVE_FAILURES = 0  # Reset on success
                if isinstance(res_json, list) and len(res_json) > 0 and "generated_text" in res_json[0]:
                    return res_json[0]["generated_text"].strip()
                elif isinstance(res_json, dict) and "generated_text" in res_json:
                    return res_json["generated_text"].strip()
            logger.warning(f"HF API returned status {res.status_code} for model {model}: {res.text}")
        except Exception as e:
            logger.warning(f"Failed to call HF Image Captioning API for model {model}: {e}")
            
    # If we get here, all models failed for this image
    CONSECUTIVE_FAILURES += 1
    if CONSECUTIVE_FAILURES >= 3:
        logger.warning("Hugging Face Image Captioning API failed 3 times consecutively. Disabling HF captioning API calls for the rest of this run to prevent timeouts.")
        HF_API_FAILED = True
        
    return default_desc

def perform_ocr_macos(image_path):
    """
    Calls the local macOS Swift OCR script to extract text from the image using native Apple Vision API.
    """
    swift_script_path = os.path.join(os.path.dirname(__file__), "ocr.swift")
    
    # Check if Swift script exists, if not write it dynamically
    if not os.path.exists(swift_script_path):
        try:
            swift_content = """import Foundation
import Vision
import AppKit

guard CommandLine.arguments.count > 1 else {
    exit(1)
}

let imagePath = CommandLine.arguments[1]
let fileURL = URL(fileURLWithPath: imagePath)

guard let image = NSImage(contentsOf: fileURL),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    exit(1)
}

let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
let request = VNRecognizeTextRequest { (request, error) in
    guard let observations = request.results as? [VNRecognizedTextObservation] else {
        return
    }
    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else { continue }
        print(candidate.string)
    }
}

request.recognitionLevel = .accurate

do {
    try requestHandler.perform([request])
} catch {
    exit(1)
}
"""
            with open(swift_script_path, "w") as f:
                f.write(swift_content)
        except Exception as e:
            logger.error(f"Failed to write Swift OCR script: {e}")
            return ""
            
    try:
        result = subprocess.run(
            ["swift", swift_script_path, image_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to perform macOS Swift OCR: {e}")
        return ""

def is_inside_table(block_bbox, table_bboxes):
    """Checks if a text block's centroid is inside any of the table bounding boxes."""
    bx0, by0, bx1, by1 = block_bbox
    cx = (bx0 + bx1) / 2
    cy = (by0 + by1) / 2
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if tx0 - 5 <= cx <= tx1 + 5 and ty0 - 5 <= cy <= ty1 + 5:
            return True
    return False

def get_page_heading(page_blocks):
    """Finds the main page heading context based on the top text blocks."""
    for b in page_blocks[:3]:
        bx0, by0, bx1, by1, text, block_no, block_type = b
        text_str = text.strip().replace("\n", " ")
        if text_str and len(text_str) < 200 and by0 < 100:
            return text_str
    return ""

def parse_pdf(pdf_path, output_image_dir, hf_token=None, chunk_size=1000, overlap=200):
    """
    Parses a PDF file and extracts text, tables, and images with layout awareness.
    Groups text by paragraph blocks, associates nearby headings and captions to tables/images,
    and returns a list of chunk dictionaries.
    """
    doc_name = os.path.basename(pdf_path)
    chunks = []
    
    # Create output image directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 1. Open PDF with PyMuPDF and pdfplumber
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path} with PyMuPDF: {e}")
        return []
        
    pdfplumber_open = False
    try:
        pdf_plumb = pdfplumber.open(pdf_path)
        pdfplumber_open = True
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path} with pdfplumber: {e}")
        
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get page blocks
        page_blocks = page.get_text("blocks")
        # Sort blocks top-to-bottom, left-to-right
        page_blocks.sort(key=lambda b: (b[1], b[0]))
        
        # Get heading context
        page_heading = get_page_heading(page_blocks)
        
        # 2. Extract Tables using pdfplumber if available
        table_bboxes = []
        if pdfplumber_open and page_num < len(pdf_plumb.pages):
            try:
                plumb_page = pdf_plumb.pages[page_num]
                tables = plumb_page.find_tables()
                
                for table_idx, t in enumerate(tables):
                    bbox = t.bbox # (x0, top, x1, bottom)
                    table_bboxes.append(bbox)
                    grid = t.extract()
                    if not grid or not any(grid):
                        continue
                        
                    # Find nearby context for this table on the page
                    tx0, ty0, tx1, ty1 = bbox
                    nearby_texts = []
                    for bx0, by0, bx1, by1, text, block_no, block_type in page_blocks:
                        cx = (bx0 + bx1) / 2
                        cy = (by0 + by1) / 2
                        if tx0 - 5 <= cx <= tx1 + 5 and ty0 - 5 <= cy <= ty1 + 5:
                            continue
                            
                        d_y = 0
                        if by1 < ty0:
                            d_y = ty0 - by1
                        elif by0 > ty1:
                            d_y = by0 - ty1
                            
                        if d_y < 80:
                            nearby_texts.append(text.strip())
                            
                    table_context = " | ".join(t.strip().replace("\n", " ") for t in nearby_texts[:3] if t.strip())
                    
                    # Split large table into row groups of 10 rows
                    header = grid[0]
                    rows = grid[1:]
                    
                    chunk_size_rows = 10
                    if len(rows) <= chunk_size_rows:
                        groups = [grid]
                    else:
                        groups = []
                        for i in range(0, len(rows), chunk_size_rows):
                            group = [header] + rows[i:i+chunk_size_rows]
                            groups.append(group)
                            
                    for group_idx, group in enumerate(groups):
                        md_table = format_table_as_markdown(group)
                        if not md_table:
                            continue
                            
                        context_prefix = f"[Document: {doc_name}, Page: {page_num + 1}, Type: table]"
                        if len(groups) > 1:
                            context_prefix += f" [Part {group_idx + 1}/{len(groups)}]"
                            
                        content_str = f"{context_prefix}\n"
                        if page_heading:
                            content_str += f"[Section/Heading: {page_heading}]\n"
                        if table_context:
                            content_str += f"[Table Caption/Context: {table_context}]\n"
                        content_str += f"{md_table}"
                        
                        chunks.append({
                            "doc_name": doc_name,
                            "page": page_num + 1,
                            "type": "table",
                            "content": content_str,
                            "metadata": {
                                "table_index": table_idx,
                                "bbox": bbox,
                                "part": group_idx + 1,
                                "total_parts": len(groups)
                            }
                        })
            except Exception as e:
                logger.error(f"Error extracting tables on page {page_num+1}: {e}")
                
        # 3. Extract Images using PyMuPDF and associate captions
        try:
            images_info = page.get_image_info(xrefs=True)
            for img_idx, img in enumerate(images_info):
                xref = img.get("xref")
                bbox = img["bbox"] # (x0, y0, x1, y1)
                
                if not xref or xref <= 0:
                    continue
                    
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                except Exception as e:
                    logger.error(f"Failed to extract image xref {xref} on page {page_num+1}: {e}")
                    continue
                    
                # Save image file
                image_filename = f"img_{doc_name.replace(' ', '_')}_p{page_num+1}_{img_idx}.{image_ext}"
                image_filepath = os.path.join(output_image_dir, image_filename)
                
                try:
                    with open(image_filepath, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    logger.error(f"Failed to write image file {image_filename}: {e}")
                    continue
                    
                # Find nearby context for this image on the page
                ix0, iy0, ix1, iy1 = bbox
                nearby_texts = []
                for bx0, by0, bx1, by1, text, block_no, block_type in page_blocks:
                    cx = (bx0 + bx1) / 2
                    cy = (by0 + by1) / 2
                    if ix0 - 5 <= cx <= ix1 + 5 and iy0 - 5 <= cy <= iy1 + 5:
                        continue
                        
                    d_y = 0
                    if by1 < iy0:
                        d_y = iy0 - by1
                    elif by0 > iy1:
                        d_y = by0 - iy1
                        
                    if d_y < 80:
                        nearby_texts.append(text.strip())
                        
                image_context = " | ".join(t.strip().replace("\n", " ") for t in nearby_texts[:3] if t.strip())
                
                # Query Hugging Face Image Captioning (Direct HTTP POST bypass)
                caption = describe_image(image_bytes, hf_token)
                
                # Perform OCR on image
                ocr_text = perform_ocr_macos(image_filepath)
                
                synonyms = "[Visual Synonyms: map, chart, graph, diagram, illustration, figure, photo, picture, image, visual]"
                
                content_str = f"[Document: {doc_name}, Page: {page_num + 1}, Type: image]\n"
                if page_heading:
                    content_str += f"[Section/Heading: {page_heading}]\n"
                if image_context:
                    content_str += f"[Image Caption/Context: {image_context}]\n"
                content_str += f"{synonyms}\n"
                content_str += f"[Image Description: {caption}]\n"
                if ocr_text:
                    content_str += f"[Image Text (OCR):\n{ocr_text}\n]"
                    
                chunks.append({
                    "doc_name": doc_name,
                    "page": page_num + 1,
                    "type": "image",
                    "content": content_str,
                    "metadata": {
                        "image_path": f"/static/images/{image_filename}",
                        "image_index": img_idx,
                        "bbox": bbox,
                        "raw_caption": caption,
                        "ocr_text": ocr_text
                    }
                })
        except Exception as e:
            logger.error(f"Error extracting images on page {page_num+1}: {e}")
            
        # 4. Extract Text and build paragraph/block-aware chunks (filtered to skip bboxes inside tables)
        try:
            filtered_blocks = []
            for b in page_blocks:
                bx0, by0, bx1, by1, text, block_no, block_type = b
                text_str = text.strip()
                if not text_str:
                    continue
                # Skip table contents
                if is_inside_table((bx0, by0, bx1, by1), table_bboxes):
                    continue
                filtered_blocks.append(text_str)
                
            # Group blocks to form chunks
            current_chunk_blocks = []
            current_length = 0
            
            for block_text in filtered_blocks:
                block_len = len(block_text)
                
                if block_len > chunk_size:
                    if current_chunk_blocks:
                        chunks.append({
                            "doc_name": doc_name,
                            "page": page_num + 1,
                            "type": "text",
                            "content": f"[Document: {doc_name}, Page: {page_num + 1}, Type: text]\n" + 
                                       (f"[Section/Heading: {page_heading}]\n" if page_heading else "") +
                                       "\n\n".join(current_chunk_blocks),
                            "metadata": {"chunk_index": len(chunks)}
                        })
                        current_chunk_blocks = []
                        current_length = 0
                        
                    sub_chunks = chunk_text(block_text, chunk_size=chunk_size, overlap=overlap)
                    for sc in sub_chunks:
                        chunks.append({
                            "doc_name": doc_name,
                            "page": page_num + 1,
                            "type": "text",
                            "content": f"[Document: {doc_name}, Page: {page_num + 1}, Type: text]\n" + 
                                       (f"[Section/Heading: {page_heading}]\n" if page_heading else "") +
                                       sc,
                            "metadata": {"chunk_index": len(chunks)}
                        })
                    continue
                    
                if current_length + block_len + 2 > chunk_size:
                    if current_chunk_blocks:
                        chunks.append({
                            "doc_name": doc_name,
                            "page": page_num + 1,
                            "type": "text",
                            "content": f"[Document: {doc_name}, Page: {page_num + 1}, Type: text]\n" + 
                                       (f"[Section/Heading: {page_heading}]\n" if page_heading else "") +
                                       "\n\n".join(current_chunk_blocks),
                            "metadata": {"chunk_index": len(chunks)}
                        })
                        
                    # Calculate overlap using entire blocks
                    overlap_blocks = []
                    overlap_len = 0
                    for prev_block in reversed(current_chunk_blocks):
                        if overlap_len + len(prev_block) + 2 <= overlap:
                            overlap_blocks.insert(0, prev_block)
                            overlap_len += len(prev_block) + 2
                        else:
                            break
                    current_chunk_blocks = overlap_blocks + [block_text]
                    current_length = sum(len(b) for b in current_chunk_blocks) + (len(current_chunk_blocks) - 1) * 2
                else:
                    current_chunk_blocks.append(block_text)
                    current_length += block_len + (2 if current_length > 0 else 0)
                    
            if current_chunk_blocks:
                chunks.append({
                    "doc_name": doc_name,
                    "page": page_num + 1,
                    "type": "text",
                    "content": f"[Document: {doc_name}, Page: {page_num + 1}, Type: text]\n" + 
                               (f"[Section/Heading: {page_heading}]\n" if page_heading else "") +
                               "\n\n".join(current_chunk_blocks),
                    "metadata": {"chunk_index": len(chunks)}
                })
        except Exception as e:
            logger.error(f"Error chunking text on page {page_num+1}: {e}")
            
    doc.close()
    if pdfplumber_open:
        pdf_plumb.close()
        
    return chunks
