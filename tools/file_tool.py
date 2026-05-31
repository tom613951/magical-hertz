import os
from langchain_core.tools import tool

# Standard absolute path of the workspace directory
WORKSPACE_DIR = r"C:\Users\26503\Documents\antigravity\magical-hertz"

@tool
def write_workspace_file(filename: str, content: str) -> str:
    """
    Write generated content (Python code, Markdown documentation, JSON/GeoJSON) 
    to a file inside the user's workspace directory. 
    Use this tool only when saving the final QA-approved GIS script or report.
    """
    # Prevent directory traversal attacks by taking the base name
    clean_filename = os.path.basename(filename)
    
    # Restrict to safe file extensions for GIS scripts and data
    allowed_extensions = {".py", ".md", ".json", ".geojson", ".html"}
    _, ext = os.path.splitext(clean_filename.lower())
    if ext not in allowed_extensions:
        return f"Error: File extension {ext} not allowed. Allowed extensions: {', '.join(allowed_extensions)}"
    
    filepath = os.path.join(WORKSPACE_DIR, clean_filename)
    try:
        # Create directories if necessary, although workspace is root
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to workspace file: {clean_filename}"
    except Exception as e:
        return f"Error writing file {clean_filename}: {str(e)}"
