from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo. Useful for looking up GIS library documentation, 
    specific Python library functions (like geopandas, shapely, folium), error messages,
    coordinate system EPSG codes, or spatial analysis algorithms.
    """
    try:
        # Use DDGS context manager to perform text search
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No search results found for: {query}"
            
            formatted_results = []
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "No description available.")
                formatted_results.append(f"[{idx}] {title}\nURL: {href}\nSnippet: {body}\n")
            
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing search query '{query}': {str(e)}"
