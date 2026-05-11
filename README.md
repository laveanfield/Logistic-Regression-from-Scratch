# Logistic Regression from Scratch

## Overview

This project presents a fully from-scratch implementation of binary Logistic Regression in Python, applied to the Breast Cancer Wisconsin (Diagnostic) dataset. Every component of the learning pipeline is implemented manually using NumPy, without relying on high-level machine learning frameworks: this includes the sigmoid activation function, the log-likelihood objective with L2 regularization, batch gradient ascent for parameter optimization, and a custom Grid Search procedure to systematically identify optimal hyperparameter configurations.

Prior to model training, an in-depth exploratory data analysis was conducted to inform key preprocessing and modeling decisions — including feature selection, handling of class imbalance, detection of multicollinearity, and the choice of feature scaling strategy. Training decisions are therefore grounded in empirical observations drawn from the data rather than applied as default conventions.

The resulting model is benchmarked against `scikit-learn`'s `LogisticRegression` to verify correctness and evaluate relative performance across standard classification metrics.

---

## Project Structure

```
Logistic-Regression-from-Scratch/
├── .streamlit/
│   └── config.toml                            # Streamlit theme configuration
├── assets/
│   ├── page_icon.png                          # App icon
│   └── style.css                              # Custom CSS for the web app
├── data/
│   ├── processed/
│   │   └── data.csv                           # Cleaned and preprocessed dataset
│   └── raw/
│       └── breast-cancer-wisconsin-data.zip   # Original raw data from Kaggle
├── models/
│   ├── lr_scratch.joblib                      # Serialized custom model (default parameters)
│   ├── lr_sklearn.joblib                      # Serialized scikit-learn model (default parameters)
│   ├── scratch_tuned.joblib                   # Serialized custom model (tuned parameters)
│   └── sklearn_tuned.joblib                   # Serialized scikit-learn model (tuned parameters)
├── src/
│   ├── config.py                              # Hyperparameters and path configurations
│   ├── data_loader.py                         # Data loading utilities
│   ├── logistic_regression_scratch.py         # Custom LogisticRegression class
│   ├── preprocessing.py                       # Feature scaling, train/test split, SMOTE
│   ├── utils.py                               # Evaluation metrics and helper functions
│   └── visualization.py                       # Loss curves, confusion matrix, ROC-AUC plots
├── threshold/
│   ├── lr_thr.joblib                          # Optimal threshold for custom model
│   └── sk_thr.joblib                          # Optimal threshold for scikit-learn model
├── app.py                                     # Streamlit web application
├── EDA_and_Preprocessing_BreastCancer.ipynb   # Exploratory data analysis and preprocessing
├── logistic_regression_breast_cancer.ipynb    # Model training, evaluation, and comparison
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## Mathematical Foundation

**Sigmoid Function**

The sigmoid function maps the linear output $z = w^T x + b$ to a probability in the interval $[0, 1]$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Log-Likelihood Objective with L2 Regularization**

Rather than minimizing a loss, the model maximizes the penalized log-likelihood of the observed labels over all $m$ training samples. An L2 regularization term is appended to discourage large weight magnitudes and reduce overfitting:

$$\ell(w, b) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] - \frac{\lambda}{2} \|w\|^2$$

where $\lambda \geq 0$ is the regularization strength. Setting $\lambda = 0$ recovers the unregularized objective.

**Batch Gradient Ascent**

At each iteration, the weights and bias are updated in the direction of the gradient of the penalized log-likelihood, computed over the entire training set:

$$w := w + \alpha \cdot \left( \frac{\partial \ell}{\partial w} - \lambda w \right), \qquad b := b + \alpha \cdot \frac{\partial \ell}{\partial b}$$

where $\alpha$ is the learning rate. The bias term $b$ is not regularized, consistent with standard practice. This ascent formulation directly reflects the maximum likelihood estimation perspective under a Gaussian prior on the weights.

---

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset consists of 569 samples described by 30 numeric features derived from digitized images of fine needle aspirate (FNA) of breast masses. The binary classification target distinguishes malignant (1) from benign (0) tumors.

| Property | Value                       |
|----------|-----------------------------|
| Samples  | 569                         |
| Features | 30 numeric                  |
| Classes  | Malignant / Benign          |
| Source   | Kaggle / `sklearn.datasets` |

---

## Implementation

The custom `LogisticRegression` class located in `src/` implements the full training pipeline from first principles:

- Sigmoid activation for mapping linear outputs to class probabilities
- Penalized log-likelihood as the optimization objective, with L2 regularization to control model complexity
- Batch gradient ascent for iterative parameter updates over the full training set, with the regularization penalty applied directly to the weight gradient
- `fit`, `predict`, and `predict_proba` methods consistent with the scikit-learn API
- A from-scratch Grid Search implementation — without external tuning libraries — that exhaustively searches over a predefined grid of learning rates, iteration counts, and regularization strengths $\lambda$
- Optional class imbalance correction via SMOTE (`imbalanced-learn`)

---

## Notebooks

**`EDA_and_Preprocessing_BreastCancer.ipynb`**

Covers feature distribution analysis, correlation heatmap, class balance assessment, feature standardization using `StandardScaler`, and train/test splitting.

**`logistic_regression_breast_cancer.ipynb`**

Covers model training with both custom and scikit-learn implementations, evaluation using accuracy, precision, recall, F1-score, and ROC-AUC, convergence analysis via loss curves, and direct performance comparison between both approaches.

---

## Results

| Model                 | Accuracy | F1 Score |
|-----------------------|----------|----------|
| Custom implementation | ~95–97%  | ~0.95+   |
| Scikit-learn baseline | ~97%     | ~0.97    |

Results may vary slightly depending on random seed, hyperparameter configuration, and whether class balancing is applied.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/laveanfield/Logistic-Regression-from-Scratch.git
cd Logistic-Regression-from-Scratch
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# On macOS / Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch Jupyter**

```bash
jupyter notebook
```

---

## Data Source Configuration

The project supports two data sources, configurable in `src/config.py`:

**Option A — Kaggle API** (downloads the raw zip automatically)

Set `source = "kaggle"` in `config.py`, then place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`. The loader will fetch the dataset and save it to `data/raw/`.

**Option B — scikit-learn** (no download required)

Set `source = "sklearn"` in `config.py`. The dataset is loaded directly via `sklearn.datasets.load_breast_cancer()` and the Kaggle API dependency is not required.

**Option C — Manual download from Kaggle website**

Set `source = "kaggle_csv"` in `config.py`. Download the dataset manually from
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data, place the
extracted `data.csv` file into `data/raw/`, then run the notebook as usual.
No Kaggle API credentials required.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib==3.9.3
seaborn
plotly
imbalanced-learn>=0.11   # optional: required only when using SMOTE
kaggle>=1.6              # optional: required only when loading data via Kaggle API
```

---

## Usage
 
```python
from src.logistic_regression_scratch import LogisticRegressionScratch
 
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000, lambda_reg=0.01)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
 
predictions  = model.predict(X_test)
probabilities = model.predict_proba(X_test)
 
model.save("models/scratch_tuned.joblib")
```
 
---
 
## Web Application
 
An interactive web application is provided via Streamlit, allowing users to explore model predictions and evaluation results without running the notebooks directly.
 
**Run the app locally**
 
```bash
streamlit run app.py
```
 
The app consists of three pages:
 
**Predict** — enter tumor measurements manually and obtain a real-time prediction from either the custom or the scikit-learn model, along with a confidence score.
 
**Compare models** — side-by-side evaluation of both models on the held-out test set, including a scatter plot of predicted probabilities to visualize agreement between the two approaches.
 
**Metrics** — detailed evaluation of the selected model, including accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

laveanfield — https://github.com/laveanfield