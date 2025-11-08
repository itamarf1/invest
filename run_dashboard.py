#!/usr/bin/env python3
"""
Investment System Web Dashboard Launcher
Run this script to start the web dashboard on http://localhost:8000
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("🚀 Starting Investment System Web Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("📈 API docs available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Run the FastAPI app
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src", "templates", "static"]
    )