from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_svm(X_train, y_train):
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    return model

def tune_svm(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, title="Evaluation"):
    preds = model.predict(X)
    print(f"\n--- {title} ---")
    print("Accuracy:", accuracy_score(y, preds))
    print("Classification Report:\n", classification_report(y, preds))
    return confusion_matrix(y, preds)

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    return scores

import joblib

__all__ = ['load_model']
def load_model(model_path="../model/vehicle_svm_model.pkl", scaler_path="../model/feature_scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler