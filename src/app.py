import os
import sys
import subprocess
import threading
import time
import webview
import socket
import logging
from datetime import datetime

# Login configuration
log_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
log_file = os.path.join(log_dir, f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DataAnalysisApp')

# Function to check if the port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Funkcja do znalezienia wolnego portu
def find_free_port():
    port = 8501  # Streamlite default port
    while is_port_in_use(port):
        port += 1
    return port

# Defining directories 
if getattr(sys, 'frozen', False):
    # If the app is packaged by PyInstaller
    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = sys._MEIPASS
# In PyInstaller mode, main.py should be in the zipped 'src' folder
    main_path = os.path.join(base_dir, 'src', 'main.py')
    logger.info(f"Tryb PyInstaller: app_dir: {app_dir}, base_dir: {base_dir}")
else:
    # In development mode
    app_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = app_dir
    # In development mode, we are already in the 'src' directory, so main.py is directly
    main_path = os.path.join(app_dir, 'main.py')
    logger.info(f"Tryb deweloperski: app_dir: {app_dir}")

logger.info(f"Ścieżka do main.py: {main_path}")

# Check if the main.py file exists
if not os.path.exists(main_path):
    logger.error(f"BŁĄD: Plik główny {main_path} nie istnieje!")
    # Checking the contents of directories
    logger.info(f"Zawartość katalogu app_dir ({app_dir}): {os.listdir(app_dir)}")
    if os.path.exists(os.path.join(app_dir, 'src')):
        logger.info(f"Zawartość katalogu src: {os.listdir(os.path.join(app_dir, 'src'))}")

# Find a free port for Streamlit
port = find_free_port()
logger.info(f"Wybrany port dla Streamlita: {port}")

# Launch Streamlit
def run_streamlit():
    try:
        if not os.path.exists(main_path):
            logger.error(f"BŁĄD: Nie można uruchomić Streamlita - plik {main_path} nie istnieje!")
            return None
        
        if getattr(sys, 'frozen', False):
            # In PyInstaller mode
            cmd = [
                sys.executable, 
                "-m", "streamlit", 
                "run", 
                main_path,
                "--server.headless", "true",
                "--server.port", str(port)
            ]
            logger.info(f"Komenda Streamlit (PyInstaller): {' '.join(cmd)}")
        else:
            # In development mode
            streamlit_cmd = os.path.join(".venv", "Scripts", "streamlit.cmd")
            cmd = [
                streamlit_cmd,
                "run",
                main_path,
                "--server.headless", "true",
                "--server.port", str(port)
            ]
            logger.info(f"Komenda Streamlit (dev): {' '.join(cmd)}")
        
        # Use subprocess instead of os.system
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Streamlite output logging
        def log_output(stream, level_func):
            for line in stream:
                level_func(f"STREAMLIT: {line.strip()}")
        
        threading.Thread(target=log_output, args=(process.stdout, logger.info), daemon=True).start()
        threading.Thread(target=log_output, args=(process.stderr, logger.error), daemon=True).start()
        
        logger.info("Proces Streamlit uruchomiony")
        return process
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania Streamlit: {str(e)}", exc_info=True)
        return None

# Run Streamlit in a separate thread
streamlit_process = None
def start_streamlit():
    global streamlit_process
    logger.info("Rozpoczynam uruchamianie Streamlit")
    streamlit_process = run_streamlit()
    if not streamlit_process:
        logger.error("Nie udało się uruchomić procesu Streamlit")

thread = threading.Thread(target=start_streamlit, daemon=True)
thread.start()
logger.info("Uruchomiono wątek Streamlit")

# Wait for Streamlit to start
def wait_for_streamlit():
    logger.info("Oczekiwanie na uruchomienie serwera Streamlit...")
    max_attempts = 30
    attempts = 0
    while attempts < max_attempts:
        if is_port_in_use(port):
            logger.info(f"Serwer Streamlit działa na porcie {port} (próba {attempts+1})")
            return True
        logger.info(f"Czekam na serwer Streamlit (próba {attempts+1}/{max_attempts})")
        time.sleep(1)
        attempts += 1
    logger.error(f"Serwer Streamlit nie uruchomił się po {max_attempts} próbach")
    return False

if not wait_for_streamlit():
    logger.error("Nie udało się uruchomić Streamlita - kończę aplikację")
    if streamlit_process:
        streamlit_process.terminate()
    sys.exit(1)

try:
    # Create application window
    logger.info(f"Tworzenie okna PyWebView dla adresu http://localhost:{port}")
    window = webview.create_window("HomeValue-Analytics", f"http://localhost:{port}")
    logger.info("Okno PyWebView utworzone")

    # Close Streamlit process after closing window
    def on_closed():
        global streamlit_process
        logger.info("Zamykanie okna aplikacji")
        if streamlit_process:
            logger.info("Kończenie procesu Streamlit")
            streamlit_process.terminate()

    window.events.closed += on_closed

    # Run the application
    logger.info("Uruchamianie PyWebView")
    webview.start()
    logger.info("PyWebView zakończył działanie")
except Exception as e:
    logger.error(f"Błąd podczas tworzenia okna PyWebView: {str(e)}", exc_info=True)
    if streamlit_process:
        streamlit_process.terminate()