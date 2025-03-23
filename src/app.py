import os
import sys
import subprocess
import threading
import time
import webview
import socket
import logging
from datetime import datetime

# Konfiguracja logowania
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

# Funkcja do sprawdzenia, czy port jest używany
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Funkcja do znalezienia wolnego portu
def find_free_port():
    port = 8501  # Domyślny port Streamlita
    while is_port_in_use(port):
        port += 1
    return port

# Określenie katalogów - uwaga: poprawiona logika ścieżek!
if getattr(sys, 'frozen', False):
    # Jeśli aplikacja jest spakowana przez PyInstaller
    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = sys._MEIPASS
    # W trybie PyInstaller, main.py powinien być w spakowanym folderze 'src'
    main_path = os.path.join(base_dir, 'src', 'main.py')
    logger.info(f"Tryb PyInstaller: app_dir: {app_dir}, base_dir: {base_dir}")
else:
    # W trybie deweloperskim
    app_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = app_dir
    # W trybie deweloperskim, jesteśmy już w katalogu 'src', więc main.py jest bezpośrednio
    main_path = os.path.join(app_dir, 'main.py')
    logger.info(f"Tryb deweloperski: app_dir: {app_dir}")

logger.info(f"Ścieżka do main.py: {main_path}")

# Sprawdź, czy plik main.py istnieje
if not os.path.exists(main_path):
    logger.error(f"BŁĄD: Plik główny {main_path} nie istnieje!")
    # Sprawdzenie zawartości katalogów
    logger.info(f"Zawartość katalogu app_dir ({app_dir}): {os.listdir(app_dir)}")
    if os.path.exists(os.path.join(app_dir, 'src')):
        logger.info(f"Zawartość katalogu src: {os.listdir(os.path.join(app_dir, 'src'))}")

# Znajdź wolny port dla Streamlita
port = find_free_port()
logger.info(f"Wybrany port dla Streamlita: {port}")

# Uruchom Streamlit
def run_streamlit():
    try:
        if not os.path.exists(main_path):
            logger.error(f"BŁĄD: Nie można uruchomić Streamlita - plik {main_path} nie istnieje!")
            return None
        
        if getattr(sys, 'frozen', False):
            # W trybie PyInstaller
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
            # W trybie deweloperskim
            streamlit_cmd = os.path.join(".venv", "Scripts", "streamlit.cmd")
            cmd = [
                streamlit_cmd,
                "run",
                main_path,
                "--server.headless", "true",
                "--server.port", str(port)
            ]
            logger.info(f"Komenda Streamlit (dev): {' '.join(cmd)}")
        
        # Użyj subprocess zamiast os.system
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Logowanie wyjścia Streamlita
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

# Uruchom Streamlit w osobnym wątku
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

# Czekaj, aż Streamlit się uruchomi
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
    # Utwórz okno aplikacji
    logger.info(f"Tworzenie okna PyWebView dla adresu http://localhost:{port}")
    window = webview.create_window("HomeValue-Analytics", f"http://localhost:{port}")
    logger.info("Okno PyWebView utworzone")

    # Zamknij proces Streamlita po zamknięciu okna
    def on_closed():
        global streamlit_process
        logger.info("Zamykanie okna aplikacji")
        if streamlit_process:
            logger.info("Kończenie procesu Streamlit")
            streamlit_process.terminate()

    window.events.closed += on_closed

    # Uruchom aplikację
    logger.info("Uruchamianie PyWebView")
    webview.start()
    logger.info("PyWebView zakończył działanie")
except Exception as e:
    logger.error(f"Błąd podczas tworzenia okna PyWebView: {str(e)}", exc_info=True)
    if streamlit_process:
        streamlit_process.terminate()