import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
API_BASE = "https://api.football-data.org/v4"
COMPETITION = "PL"