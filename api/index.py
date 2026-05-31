from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response_data = {
            "project": "magical-hertz",
            "description": "Magical Hertz GIS Multi-Agent & Advanced RAG System",
            "modules": {
                "geograph": "LangGraph GIS Developer simulation. Run locally using: streamlit run app.py",
                "advanced_rag": "Hybrid BM25 + Vector search with Flashrank reranking. Run locally using: streamlit run rag_system/app.py"
            },
            "status": "online",
            "author": "tom613951"
        }
        
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
        return
