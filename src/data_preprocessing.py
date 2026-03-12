import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import DATA_PATH, SENSOR_COLS, TARGET_COL, OUTLIER_THRESHOLDS, PLOTS_DIR


def load_data(path=DATA_PATH):
    data = pd.read_csv(path, dtype="unicode")
    cols = data.columns
    data[cols[1:]] = data[cols[1:]].apply(pd.to_numeric, errors="coerce")
    return data


def drop_all_zero_rows(df):
    return df[~(df[SENSOR_COLS] == 0).all(axis=1)]


def build_scaling_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("std_scaler", StandardScaler()),
    ])


def scale_features(df):
    features = df[SENSOR_COLS]
    labels = df[TARGET_COL]

    pipeline = build_scaling_pipeline()
    scaled = pipeline.fit_transform(features)

    scaled_df = pd.DataFrame(scaled, columns=SENSOR_COLS, index=features.index)
    return pd.concat([scaled_df, labels], axis=1), pipeline


def remove_outliers(df):
    cleaned = df.copy()
    for sensor, threshold in OUTLIER_THRESHOLDS.items():
        outlier_idx = cleaned[cleaned[sensor] > threshold].index
        cleaned = cleaned.drop(outlier_idx)
    return cleaned


def plot_sensor_data(df, save_dir=PLOTS_DIR):
    """
    Notebook: df.plot('Time_Stamp', 'SensorX') for each sensor.
    Uses pandas .plot() directly — same as notebook — to correctly
    handle string Time_Stamp on x-axis with 65k points.
    Saves one PNG per sensor into results/plots/.
    """
    os.makedirs(save_dir, exist_ok=True)
    for sensor in SENSOR_COLS:
        ax = df.plot(x="Time_Stamp", y=sensor, figsize=(10, 4), legend=True)
        ax.set_xlabel("Time_Stamp")
        ax.set_ylabel(sensor)
        ax.set_title(f"{sensor} — Vibration over Time")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{sensor.lower()}_timeseries.png")
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  Saved → {path}")


def plot_class_distribution(labels, save_dir=PLOTS_DIR, tag=""):
    """
    Notebook: plt.pie(df['Fault_Detection'].value_counts()...) — raw and post-cleaning.
    Saves pie chart PNG into results/plots/.
    """
    os.makedirs(save_dir, exist_ok=True)
    counts = labels.value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(
        counts.values,
        labels=["Faulty", "Non-Faulty"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
    )
    title = f"Class Distribution{' — ' + tag if tag else ''}"
    plt.title(title)
    plt.tight_layout()
    filename = f"class_distribution{'_' + tag if tag else ''}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved → {path}")


def preprocess(path=DATA_PATH):
    data = load_data(path)
    df = pd.DataFrame(data, columns=["Time_Stamp"] + SENSOR_COLS + [TARGET_COL])

    df = drop_all_zero_rows(df)
    scaled_df, pipeline = scale_features(df)
    cleaned_df = remove_outliers(scaled_df)

    # Shuffle
    dataset = cleaned_df.sample(frac=1, random_state=42)
    features = dataset[SENSOR_COLS]
    labels = dataset[TARGET_COL]

    return features, labels, pipeline
