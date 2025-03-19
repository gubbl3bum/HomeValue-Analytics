import os
import threading
import webview
import subprocess

# Run strimlit
def run_streamlit():
    os.system(".\.venv\Scripts\streamlit.cmd run .\src\main.py --server.headless true")

# Run strimlit in separate thread
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Create desktop window
webview.create_window("Analiza Danych", "http://localhost:8501")
webview.start()
