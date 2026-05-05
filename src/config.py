from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#===========================================
# Dataset
#===========================================

CSV_FILENAME = "data.csv"
CSV_PATH = PROCESSED_DIR / CSV_FILENAME

KAGGLE_ZIP_NAME = "breast-cancer-wisconsin-data.zip"
KAGGLE_ZIP_PATH = RAW_DIR / KAGGLE_ZIP_NAME
KAGGLE_DATASET_SLUG = "uciml/breast-cancer-wisconsin-data"

#===========================================
# Handling columns
#===========================================

DROP_COLS = ["id", "Unnamed: 32"]
TARGET_COL = "diagnosis"
ID_COL = "id"


#===========================================
# Params in preprocessing
#===========================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.125
SMOTE_STRATEGY = 1.0

#===========================================
# Color
#===========================================

COLORS = {
    "malignant" : "#FF5851",
    "benign"    : "#3d9dfc",
    "purple"    : "#d852fa",
    "dark_gray" : "#494a49",
    "black"     : "#212121",
    "white"     : "#ffffff",
    "gray"      : "#727372",
    "green"     : "#03fc39",
    "teal"      : "#00b4d8",
    "orange"    : "#f4a261",
}

COLOR_LIST = list(COLORS.values())

#===========================================
# Make sure directories is available
#===========================================

for _dir in (RAW_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
