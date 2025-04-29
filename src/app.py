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
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)

# Streamlit port configuration
STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# Path handling for both development and frozen environments
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_EXE = os.path.join(BASE_DIR, "streamlit.exe")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_EXE = "streamlit"

# Function to run Streamlit
def run_streamlit():
    try:
        logging.info(f"Starting Streamlit server with script: {MAIN_SCRIPT}")
        
        if getattr(sys, 'frozen', False):
            cmd = [STREAMLIT_EXE, "run", MAIN_SCRIPT, 
                  "--server.port", str(STREAMLIT_PORT), 
                  "--server.headless", "true"]
        else:
            cmd = ["streamlit", "run", MAIN_SCRIPT, 
                  "--server.port", str(STREAMLIT_PORT), 
                  "--server.headless", "true"]
        
        # Check if file exists
        if not os.path.exists(MAIN_SCRIPT):
            logging.error(f"Main script not found at: {MAIN_SCRIPT}")
            return
            
        logging.info(f"Running command: {' '.join(cmd)}")
        
        # ZMIANA TUTAJ: Usunięto shell=True i dodano creationflags
        startupinfo = None
        if sys.platform == 'win32':
            # Ukryj okno konsoli na Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW  # Dodatkowa flaga dla ukrycia okna konsoli
            )
        else:
            # Na innych systemach operacyjnych
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        # Monitor output
        for line in process.stdout:
            logging.info(f"STREAMLIT: {line.strip()}")
        for line in process.stderr:
            logging.error(f"STREAMLIT ERROR: {line.strip()}")
            
    except Exception as e:
        logging.error(f"Error starting Streamlit: {str(e)}")
        logging.exception("Full traceback:")

# Start Streamlit in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

# Waiting for Streamlit server to start
logging.info("Waiting for Streamlit server to start...")
time.sleep(5)

# Start PyWebView
logging.info(f"Starting PyWebView with URL {STREAMLIT_URL}")
webview.create_window("HomeValue-Analytics", STREAMLIT_URL)
webview.start()