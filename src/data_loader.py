import zipfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.config import (
    CSV_PATH,
    PROCESSED_DIR,
    KAGGLE_ZIP_PATH,
    KAGGLE_DATASET_SLUG,
    TARGET_COL,
    DROP_COLS

)

class DataLoader:
    def load(self, source: str="sklearn", force_extract: bool = False) -> pd.DataFrame:
        if source == "sklearn":
            return self._load_from_sklearn()
        
        if source == "csv":
            if CSV_PATH.exists() and not force_extract:
                print(f"Read CSV from: {CSV_PATH}...")
                return self._read_csv(CSV_PATH)
            elif KAGGLE_ZIP_PATH.exists():
                print(f"Extracting ZIP...")
                self._extract_zip(KAGGLE_ZIP_PATH)
                return self._read_csv(CSV_PATH)
            else:
                print(f"CSV not found!")
                return self._load_from_sklearn()

        if source == "kaggle":
            self._download_via_kaggle_api()
            return self._read_csv(CSV_PATH)
        
        raise ValueError(f"Source must be 'sklearn', 'csv', 'kaggle'.")
            
    #===========================================
    # Helper function
    #===========================================

    def _load_from_sklearn(self) -> pd.DataFrame:
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)

        df[TARGET_COL] = pd.Categorical(
            np.where(raw.target == 0, "M", "B"),
            categories=["M", "B"]
        )

        cols = [TARGET_COL] + [c for c in df.columns if c != TARGET_COL]
        df = df[cols].reset_index(drop=True)
        print("Loading completed from sklearn.")

        return df


    def _read_csv(self, csv_path: Path = CSV_PATH) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
        print(f"Have read {df.shape[0]} feartures {df.shape[1]} columns")
        return df
        
    def _extract_zip(self, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            target = "data.csv"
            if target not in zf.namelist():
                raise FileNotFoundError(f"Cannot find {target} in {zip_path}")

            zf.extract(target, PROCESSED_DIR)

    def _download_via_kaggle_api(self) -> None:
        try:
            subprocess.run(
                [
                "kaggle", "datasets", "download", "-d", KAGGLE_DATASET_SLUG, "-p", str(KAGGLE_ZIP_PATH.parent)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            zip_files = list(KAGGLE_ZIP_PATH.parent.glob("*.zip"))
            if not zip_files:
                raise FileNotFoundError("Kaggle download completed but no zip file was found.")
            self._extract_zip(zip_files[0])
        except FileNotFoundError:
            raise EnvironmentError(
                "Kaggle CLI is not installed.\n"
                "Install it using: pip install kaggle\n"
                "Place the kaggle.json file in ~/.kaggle/\n"
            )
        except subprocess.CalledProcessError as e:
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise RuntimeError(f"Kaggle CLI failded: {e}")
        
#
def load_breast_cancer_data(source: str = "sklearn") -> pd.DataFrame:
    return DataLoader().load(source=source)