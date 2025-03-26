import os
import time
import threading
import webview
import subprocess
import logging
import requests
from datetime import datetime

# Create a directory for logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a dynamic log file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"HomeValue-Analytics_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Streamlit server configuration
STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# Function to start the Streamlit server in a separate thread
def run_streamlit():
    logging.info("Starting Streamlit server...")
    cmd = ["streamlit", "run", "src/main.py", "--server.port", str(STREAMLIT_PORT), "--server.headless", "true"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Capture Streamlit logs
    for line in process.stdout:
        logging.info(f"STREAMLIT: {line.strip()}")  # Log Streamlit output

# Start Streamlit in a separate daemon thread
threading.Thread(target=run_streamlit, daemon=True).start()

# Function to wait for the Streamlit server to become available
def wait_for_streamlit(max_retries=10, delay=1):
    logging.info("Waiting for Streamlit server to start...")
    for attempt in range(max_retries):
        try:
            response = requests.get(STREAMLIT_URL)
            if response.status_code == 200:
                logging.info("Streamlit server is ready!")
                return True
        except requests.ConnectionError:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Streamlit is not available yet...")
        time.sleep(delay)
    
    logging.error("Failed to connect to Streamlit server after 10 attempts!")
    return False

# Wait for Streamlit to start before launching PyWebView
if wait_for_streamlit():
    logging.info(f"Launching PyWebView with URL: {STREAMLIT_URL}")
    webview.create_window("HomeValue-Analytics", STREAMLIT_URL)
    webview.start()
else:
    logging.error("Error: Cannot start PyWebView because Streamlit is not running!")
