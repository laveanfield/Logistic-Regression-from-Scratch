import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

from src.config import (
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    SMOTE_STRATEGY
)

class BreastCancerPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.fearture_names = []

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fearture_names = list(X_train.columns)
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)
    
    def encode_labels(self, y: pd.Series) -> np.ndarray:
        return self.le.fit_transform(y)
    
def preprocess_breast_cancer_data(
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], BreastCancerPreprocessor]:
    preprocessor = BreastCancerPreprocessor()

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    y = preprocessor.encode_labels(y_raw)
    feature_names = list(X.columns)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, 
        test_size=test_size, random_state=random_state, 
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval
    )

    X_train_sc = preprocessor.fit_transform(X_train)
    X_val_sc = preprocessor.transform(X_val)
    X_test_sc = preprocessor.transform(X_test)

    print("After scaling:")
    print(f"X_train : {X_train_sc.shape}  | Malignant: {y_train.sum()} / {len(y_train)}")
    print(f"X_val   : {X_val_sc.shape}   | Malignant: {y_val.sum()} / {len(y_val)}")
    print(f"X_test  : {X_test_sc.shape}  | Malignant: {y_test.sum()} / {len(y_test)}")

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, feature_names, preprocessor

# Add bias to each sample (x_0 = 1)
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

# We dont apply the function below to our project, I wrote the function for my own purpose only.

def apply_smote(
        X_train: np.ndarray,
        y_train: np.ndarray,
        strategy: float = SMOTE_STRATEGY, 
        random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("Must install imbalanced-learn: pip install imbalanced-learn.")
    
    sm = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {len(y_train)} samples -> After SMOTE: {len(y_res)} samples")
    print(f"Malignant: {(y_res == 1).sum()}, Benign: {(y_res==0).sum()}")

    return X_res, y_res