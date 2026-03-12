# 🔧 Fault Detection & Prediction in Centrifugal Pumps

> **End-to-end machine learning pipeline** to detect faults in industrial centrifugal pumps using real-time vibration sensor data covering data cleaning, feature scaling, model training, hyperparameter tuning, and evaluation.

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)

---

## 📌 Problem Statement

Centrifugal pumps are critical components in industrial systems. Undetected faults lead to equipment failure and costly downtime. This project builds a **binary fault classifier** "*Faulty vs Non-Faulty*" from vibration data collected across 6 sensors, enabling predictive maintenance.

---

## 📊 Dataset

| Property         | Value                              |
|------------------|------------------------------------|
| Total records    | 65,112                             |
| Features         | 6 vibration sensors                |
| Target           | `1` = Faulty, `0` = Non-Faulty     |
| Source file      | `data/Vibration_Data_New.csv`      |

---

## 🗂️ Project Structure

```
📦 Fault_detection_and_prediction_in_centrifugal_pumps
├── data
│   └── Vibration_Data_New.csv       # Raw vibration sensor data
├── models/                          # Saved trained models (.pkl)
├── notebook/                        # Exploratory analysis notebook
├── results/
│   ├── metrics_report.txt            # Before/after tuning accuracy report
│   └── plots/                        # All saved plots (sensor, pie, confusion matrix)
├── src/
│   ├── data_preprocessing.py         # Load, clean, scale, plot
│   ├── train.py                      # Train & Tune Models
│   ├── evaluate.py                   # Metrics, Confusion Matrix, Comparison
│   └── predict.py                    # Single & batch inference
├── tests/
│   └── test_preprocessing.py
├── config.py                         # Paths, hyperparameters, constants
├── main.py                           # End-to-end pipeline entry point
└── requirements.txt
```

---

## ⚙️ Pipeline

```
Raw CSV  →  Clean  →  Scale  →  Remove Outliers  →  Split  →  Train  →  Tune  →  Evaluate
```

| Step | Description |
|------|-------------|
| 🧹 **Clean** | Drop rows where all 6 sensor values are zero |
| 📐 **Scale** | `StandardScaler` via sklearn `Pipeline` with mean imputation |
| 📉 **Outliers** | Per-sensor threshold filtering (post-scaling) |
| 🔀 **Split** | 75% train / 25% test — stratified to preserve class balance |
| 🤖 **Train** | Default params first, then `GridSearchCV` hyperparameter tuning |
| 📊 **Evaluate** | Accuracy, classification report, confusion matrix, before/after comparison |

---

## 🤖 Models & Results

### Supervised — Before vs After Hyperparameter Tuning

| Model               | Default Accuracy | Tuned Accuracy | Gain     |
|---------------------|-----------------|----------------|----------|
| Logistic Regression | 75.88%          | 75.96%         | +0.08%   |
| SVM                 | 75.80%          | **78.80%**     | +3.00%   |
| KNN                 | 75.04%          | 77.01%         | +1.97%   |

### Best Hyperparameters (GridSearchCV)

| Model               | Best Params                                        |
|---------------------|----------------------------------------------------|
| Logistic Regression | `C=0.01`, `penalty='l1'`, `solver='liblinear'`     |
| SVM                 | `C=1`, `kernel='rbf'`, `gamma='scale'`             |
| KNN                 | `n_neighbors=11`, `metric='euclidean'`             |

### Unsupervised

| Model   | Result                                          |
|---------|-------------------------------------------------|
| K-Means | 3 clusters — Silhouette Score: **0.234**        |
| DBSCAN  | 46 clusters — Silhouette Score: **-0.292**      |

### 🏆 Best Model
**SVM with RBF kernel** (`C=1`, `gamma='scale'`) — Test Accuracy: **78.80%**

---

## 📈 Output Plots

All plots are auto-saved to `results/plots/` when you run the pipeline:

| Plot | Description |
|------|-------------|
| `sensor1_timeseries.png` … `sensor6_timeseries.png` | Vibration signal over time per sensor |
| `class_distribution_raw.png` | Faulty vs Non-Faulty split in raw data |
| `class_distribution_after_cleaning.png` | Class balance after preprocessing |
| `svm_tuned_rbf_kernel_confusion_matrix.png` | Confusion matrix of best model |

---

## 🚀 Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/Diksha-Bidikar/Fault_detection_and_prediction_in_centrifugal_pumps.git
cd Fault_detection_and_prediction_in_centrifugal_pumps
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the full pipeline
```
python main.py
```

### 4. Predict on new sensor readings
```python
from src.predict import predict_single

# Pass 6 pre-scaled sensor values [S1, S2, S3, S4, S5, S6]
result = predict_single([0.5, 1.2, -0.3, 0.8, 1.1, -0.5])
print(result)  # 'Faulty' or 'Non-Faulty'
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white) | Core language |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-orange?logo=scikit-learn&logoColor=white) | ML models, pipelines, GridSearchCV |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-blue) | Visualisation |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-76b7b2) | Statistical plots |

## 👤 Author
Diksha Bidikar, Master's in Data Science Rice University


LinkedIn: linkedin.com/in/diksha-bidikar


Portfolio: https://diksha-bidikar.github.io/


Email: bidikar.diksha@gmail.com