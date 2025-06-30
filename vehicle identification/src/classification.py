import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV

# ─── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "Data")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Filenames for your splits
TRAIN_X = os.path.join(DATA_DIR, "X_train_combined.npy")
VAL_X   = os.path.join(DATA_DIR, "X_val_combined.npy")
TEST_X  = os.path.join(DATA_DIR, "X_test_combined.npy")
TRAIN_Y = os.path.join(DATA_DIR, "y_train_combined.npy")
VAL_Y   = os.path.join(DATA_DIR, "y_val_combined.npy")
TEST_Y  = os.path.join(DATA_DIR, "y_test_combined.npy")

# ─── Helpers ────────────────────────────────────────────────────────────────
def load_splits():
    X_train = np.load(TRAIN_X)
    X_val   = np.load(VAL_X)
    X_test  = np.load(TEST_X)
    y_train = np.load(TRAIN_Y)
    y_val   = np.load(VAL_Y)
    y_test  = np.load(TEST_Y)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_random_forest(X_train, y_train, **rf_kwargs):
    """
    Train a Random Forest with given keyword args.
    Returns the trained model.
    """
    rf = RandomForestClassifier(**rf_kwargs, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def tune_hyperparameters(X_train, y_train):
    """
    Perform a quick grid search over n_estimators and max_depth.
    Returns the best estimator.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
    }
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        base_rf, param_grid,
        cv=3, scoring="accuracy",
        verbose=2, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X, y, label="Validation"):
    """
    Print accuracy, classification report, and return confusion matrix.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"--- {label} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    return cm

def save_model(model, name="rf_combined.pkl"):
    """
    Persist the trained model to disk.
    """
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

# ─── Main Script ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # 2) (Optional) Hyperparameter tuning
    best_rf, best_params = tune_hyperparameters(X_train, y_train)
    print("Best RF params:", best_params)

    # 3) Evaluate on validation
    cm_val = evaluate_model(best_rf, X_val, y_val, label="Validation")

    # 4) Retrain on train+val for final model
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    final_rf = train_random_forest(X_comb, y_comb, **best_params)

    # 5) Evaluate on test set
    cm_test = evaluate_model(final_rf, X_test, y_test, label="Test")

    # 6) Save final model
    save_model(final_rf, name="rf_combined_final.pkl")