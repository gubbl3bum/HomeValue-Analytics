import os
import time
import threading
import webview
import subprocess
import logging
from datetime import datetime

# Creating logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generating dynamic log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"HomeValue-Analytics_{timestamp}.log")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)

# Streamlit port configuration
STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# Function to run Streamlit server in a separate thread
def run_streamlit():
    logging.info("Starting Streamlit server...")
    cmd = ["streamlit", "run", "src/main.py", "--server.port", str(STREAMLIT_PORT), "--server.headless", "true"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        logging.info(f"STREAMLIT: {line.strip()}")  # Logging Streamlit output

# Start Streamlit in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

# # Waiting for Streamlit server to start
logging.info("Waiting for Streamlit server to start...")
time.sleep(5)

# Start PyWebView
logging.info(f"Starting PyWebView with URL {STREAMLIT_URL}")
webview.create_window("HomeValue-Analytics", STREAMLIT_URL)
webview.start()
