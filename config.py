from pathlib import Path

# Usar __file__ para rutas absolutas
PROJECT_ROOT = Path(__file__).resolve().parent

# Rutas
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "dataset_algoritmia" / "train_dataset_clean.jsonl"
DB_PATH = DATA_DIR / "chat_history.db"
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTER_PATH = MODELS_DIR / "qwen-algo-tutor-1.5b-v2"

# Configuración
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
STREAMLIT_PORT = 8501
STREAMLIT_ADDRESS = "0.0.0.0"
STREAMLIT_THEME = "dark"
COMPILER_TIMEOUT = 10
COMPILER_MAX_OUTPUT = 10000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.py', '.java', '.cpp', '.c', '.js', '.go', '.rs', '.cs'}