"""
Streamlit UI Launcher.
Starts the modern conversational frontend on http://localhost:8501.
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "main.py")
    print(f"🚀 Launching Streamlit UI ({ui_path})...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
