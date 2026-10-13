"""
FastAPI Server Launcher.
Runs the Uvicorn ASGI server on http://localhost:8000.
"""
import os
import sys
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Finance Market Intelligence API on http://localhost:8000 ...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
