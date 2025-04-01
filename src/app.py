import os
import sys
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

if getattr(sys, 'frozen', False):  
    BASE_DIR = sys._MEIPASS
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_PATH = os.path.join(BASE_DIR, "streamlit.exe")  # Adjust for frozen mode
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_PATH = os.path.join(BASE_DIR, ".venv", "Scripts", "streamlit.exe")

# Function to run Streamlit
def run_streamlit():
    logging.info(f"Starting Streamlit server with script: {MAIN_SCRIPT}")
    logging.info(f"Using Streamlit executable at: {STREAMLIT_PATH}")

    if not os.path.exists(STREAMLIT_PATH):
        logging.error(f"Streamlit executable not found: {STREAMLIT_PATH}")
        return

    if not os.path.exists(MAIN_SCRIPT):
        logging.error(f"Streamlit script not found: {MAIN_SCRIPT}")
        return

    cmd = [STREAMLIT_PATH, "run", MAIN_SCRIPT, "--server.port", str(STREAMLIT_PORT), "--server.headless", "true"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        logging.info(f"STREAMLIT: {line.strip()}")
    for line in process.stderr:
        logging.error(f"STREAMLIT ERROR: {line.strip()}")

# Start Streamlit in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

# Waiting for Streamlit server to start
logging.info("Waiting for Streamlit server to start...")
time.sleep(5)

# Start PyWebView
logging.info(f"Starting PyWebView with URL {STREAMLIT_URL}")
webview.create_window("HomeValue-Analytics", STREAMLIT_URL)
webview.start()
