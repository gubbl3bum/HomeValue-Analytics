import os
import sys
import time
import threading
import webview
import subprocess
import logging
import signal
import atexit
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

# Global variable to store the Streamlit process
streamlit_process = None

# Path handling for both development and frozen environments
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_EXE = os.path.join(BASE_DIR, "streamlit.exe")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")
    STREAMLIT_EXE = "streamlit"

# Function to terminate Streamlit process
def cleanup_streamlit():
    global streamlit_process
    if streamlit_process:
        logging.info("Terminating Streamlit process...")
        try:
            # Kill the Streamlit process properly
            if sys.platform == "win32":
                # On Windows, use taskkill to ensure all child processes are terminated
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(streamlit_process.pid)], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            else:
                # On Unix-like systems
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
            
            streamlit_process = None
            logging.info("Streamlit process terminated successfully")
        except Exception as e:
            logging.error(f"Error terminating Streamlit process: {str(e)}")

# Function to run Streamlit
def run_streamlit():
    global streamlit_process
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
        
        # Start the process with proper setup for cleanup
        if sys.platform == "win32":
            # For Windows
            streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # For Unix-like systems
            streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                preexec_fn=os.setsid
            )
        
        # Monitor output
        for line in streamlit_process.stdout:
            logging.info(f"STREAMLIT: {line.strip()}")
        for line in streamlit_process.stderr:
            logging.error(f"STREAMLIT ERROR: {line.strip()}")
            
    except Exception as e:
        logging.error(f"Error starting Streamlit: {str(e)}")
        logging.exception("Full traceback:")

# Register cleanup function for normal exit
atexit.register(cleanup_streamlit)

# Define window close event handler
def on_closed():
    logging.info("Window closed, cleaning up...")
    cleanup_streamlit()
    logging.info("Application shutdown complete")

# Start Streamlit in a separate thread
streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
streamlit_thread.start()

# Waiting for Streamlit server to start
logging.info("Waiting for Streamlit server to start...")
time.sleep(5)

# Start PyWebView with window close handler
logging.info(f"Starting PyWebView with URL {STREAMLIT_URL}")
window = webview.create_window("HomeValue-Analytics", STREAMLIT_URL)
window.events.closed += on_closed
webview.start()

# Final cleanup in case the above doesn't trigger
cleanup_streamlit()