import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from config import (
    MODEL_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    LR_DEFAULT_PARAMS,
    SVM_DEFAULT_PARAMS,
    KNN_DEFAULT_PARAMS,
    LR_PARAM_GRID,
    SVM_PARAM_GRID,
    KNN_PARAM_GRID,
    KMEANS_PARAMS,
    DBSCAN_PARAMS,
)


def split_data(features, labels):
    return train_test_split(
        features, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


# ------------------------------------------------------------------
# Default models (before hyperparameter tuning)
# ------------------------------------------------------------------

def train_logistic_regression_default(X_train, y_train):
    model = LogisticRegression(**LR_DEFAULT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_svm_default(X_train, y_train):
    model = SVC(**SVM_DEFAULT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_knn_default(X_train, y_train):
    model = KNeighborsClassifier(**KNN_DEFAULT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_train):
    model = KMeans(**KMEANS_PARAMS, random_state=RANDOM_STATE, n_init=10)
    model.fit(X_train)
    return model


def train_dbscan(X_train):
    model = DBSCAN(**DBSCAN_PARAMS)
    model.fit(X_train)
    return model


# ------------------------------------------------------------------
# Hyperparameter tuning with GridSearchCV
# ------------------------------------------------------------------

def tune_logistic_regression(X_train, y_train):
    """
    Notebook: StratifiedKFold(n_splits=5) + GridSearchCV
    Best found: C=0.01, penalty='l1', solver='liblinear'
    """
    log_reg = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(log_reg, LR_PARAM_GRID, cv=skf, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Validation Accuracy: {grid_search.best_score_*100:.2f}%")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_svm(X_train, y_train):
    """
    Notebook: further split train → trainsplit/val, then GridSearchCV cv=5
    Best found: C=1, kernel='rbf', gamma='scale'
    Grid is focused on linear & rbf only (poly/sigmoid are very slow on large data).
    n_jobs=-1 uses all CPU cores to parallelise fits.
    """
    X_trainsplit, X_val, y_trainsplit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    # Reduced grid: linear + rbf cover the notebook's best result (rbf, C=1, scale)
    svm_param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    svc = SVC()
    grid_search = GridSearchCV(estimator=svc, param_grid=svm_param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_trainsplit, y_trainsplit)

    val_acc = grid_search.score(X_val, y_val)
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    return grid_search.best_estimator_, grid_search.best_params_, val_acc


def tune_knn(X_train, y_train):
    """
    Notebook: GridSearchCV cv=5
    Best found: n_neighbors=11, metric='euclidean'
    """
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, KNN_PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Validation Accuracy: {grid_search.best_score_*100:.2f}%")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ------------------------------------------------------------------
# Model persistence
# ------------------------------------------------------------------

def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)
