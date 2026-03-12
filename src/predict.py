import numpy as np
import pandas as pd

from config import SENSOR_COLS
from src.train import load_model


def predict_single(sensor_values: list, model_filename: str = "svm_model.pkl"):
    """
    Predict fault for a single reading.

    Args:
        sensor_values: list of 6 floats [S1, S2, S3, S4, S5, S6] (pre-scaled)
        model_filename: saved model file in models/

    Returns:
        'Faulty' or 'Non-Faulty'
    """
    model = load_model(model_filename)
    X = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(X)[0]
    return "Faulty" if prediction == 1 else "Non-Faulty"


def predict_batch(df: pd.DataFrame, model_filename: str = "svm_model.pkl"):
    """
    Predict faults for a DataFrame of sensor readings.

    Args:
        df: DataFrame with columns matching SENSOR_COLS (pre-scaled)
        model_filename: saved model file in models/

    Returns:
        numpy array of predictions
    """
    model = load_model(model_filename)
    X = df[SENSOR_COLS].values
    predictions = model.predict(X)
    return np.where(predictions == 1, "Faulty", "Non-Faulty")
