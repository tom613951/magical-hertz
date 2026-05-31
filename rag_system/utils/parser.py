import os
from pypdf import PdfReader
from langchain_core.documents import Document

def parse_file(file_bytes, filename: str) -> str:
    """Extract raw text from file bytes based on the file extension."""
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == ".pdf":
        try:
            reader = PdfReader(file_bytes)
            text_parts = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file '{filename}': {str(e)}")
            
    elif ext in [".txt", ".md", ".json", ".geojson", ".html"]:
        try:
            # Decode file bytes to string
            return file_bytes.read().decode("utf-8", errors="ignore")
        except AttributeError:
            # If file_bytes is a file-like object and doesn't have read (like local files)
            with open(file_bytes, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read text file '{filename}': {str(e)}")
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only PDF, TXT, MD, JSON/GeoJSON are supported.")

def chunk_text(text: str, filename: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    """
    Split text into overlapping chunks and wrap them into LangChain Document objects.
    Ensures that we retain the source filename and chunk index in metadata.
    """
    if not text.strip():
        return []
        
    words = text.split()
    chunks = []
    
    # We do a simple word-based splitting or character-based splitting.
    # Character-based splitting is standard. Let's do character-based sliding window.
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        # Avoid orphan characters by trying to snap to nearest whitespace if possible
        if end < len(text):
            last_space = chunk_content.rfind(' ')
            if last_space > chunk_size // 2:  # If we found space in last half of chunk
                end = start + last_space
                chunk_content = text[start:end]
                
        chunks.append(Document(
            page_content=chunk_content.strip(),
            metadata={
                "source": filename,
                "chunk_id": chunk_idx,
                "start_char": start,
                "end_char": end
            }
        ))
        
        start += (chunk_size - chunk_overlap)
        chunk_idx += 1
        
    return chunks
