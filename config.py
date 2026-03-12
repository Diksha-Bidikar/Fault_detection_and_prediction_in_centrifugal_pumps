import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Vibration_Data_New.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "results", "plots")

# Feature columns
SENSOR_COLS = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5", "Sensor6"]
TARGET_COL = "Fault_Detection"

# Outlier thresholds (in scaled units, post StandardScaler)
OUTLIER_THRESHOLDS = {
    "Sensor1": 10,
    "Sensor2": 10,
    "Sensor3": 5,
    "Sensor4": 12,
    "Sensor5": 10,
    "Sensor6": 6,
}

# Train/test split
TEST_SIZE = 0.25
RANDOM_STATE = 10

# ---------------------------------------------------------
# Default hyperparameters (before tuning)
# ---------------------------------------------------------
LR_DEFAULT_PARAMS = {"C": 0.1, "penalty": "l2", "max_iter": 1000}
SVM_DEFAULT_PARAMS = {"kernel": "linear", "C": 1.0}
KNN_DEFAULT_PARAMS = {"n_neighbors": 3}

# ---------------------------------------------------------
# Hyperparameter grids for GridSearchCV
# ---------------------------------------------------------
LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}

SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"],
}

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "metric": ["euclidean", "manhattan"],
}

# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------
KMEANS_PARAMS = {"n_clusters": 3}
DBSCAN_PARAMS = {"eps": 0.5, "min_samples": 10}
