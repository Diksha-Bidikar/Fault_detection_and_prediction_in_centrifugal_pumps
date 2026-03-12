import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import PLOTS_DIR, RESULTS_DIR


def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-Faulty", "Faulty"])
    print(f"\n{'='*40}")
    print(f"{model_name}")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(report)
    return acc, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model", save_dir=PLOTS_DIR):
    """Plots and saves confusion matrix to results/plots/."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Faulty", "Faulty"],
                yticklabels=["Non-Faulty", "Faulty"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_confusion_matrix.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved → {path}")


def evaluate_clustering(X, labels, model_name="Clustering Model"):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"\n{model_name}")
    print(f"  Estimated number of clusters : {n_clusters}")
    print(f"  Estimated number of noise pts: {n_noise}")
    unique = set(labels)
    if len(unique) > 1:
        score = silhouette_score(X, labels)
        print(f"  Silhouette Score             : {score:.4f}")
        return score
    else:
        print("  Silhouette Score: N/A (only one cluster found)")
        return None


def compare_models(results: dict, title="Model Comparison"):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for name, acc in results.items():
        print(f"  {name:<30} {acc*100:.2f}%")
    best = max(results, key=results.get)
    print(f"\n  Best model: {best} ({results[best]*100:.2f}%)")


def compare_before_after(before: dict, after: dict):
    print("\n" + "="*60)
    print("Before vs After Hyperparameter Tuning")
    print("="*60)
    print(f"  {'Model':<30} {'Before':>10} {'After':>10} {'Gain':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for name in before:
        b = before[name] * 100
        a = after.get(name, 0) * 100
        gain = a - b
        sign = "+" if gain >= 0 else ""
        print(f"  {name:<30} {b:>9.2f}% {a:>9.2f}% {sign}{gain:>8.2f}%")
    print()


def save_metrics_report(before: dict, after: dict, clustering: dict, best_params: dict,
                        save_dir=RESULTS_DIR):
    """Saves a formatted text report of all results to results/metrics_report.txt."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Fault Detection in Centrifugal Pumps — Results Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUPERVISED MODELS — Before vs After Hyperparameter Tuning\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'Model':<30} {'Before':>10} {'After':>10} {'Gain':>10}\n")
        f.write(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}\n")
        for name in before:
            b = before[name] * 100
            a = after.get(name, 0) * 100
            gain = a - b
            sign = "+" if gain >= 0 else ""
            f.write(f"  {name:<30} {b:>9.2f}% {a:>9.2f}% {sign}{gain:>8.2f}%\n")

        f.write("\n\nBEST HYPERPARAMETERS (after tuning)\n")
        f.write("-" * 60 + "\n")
        for model, params in best_params.items():
            f.write(f"  {model}: {params}\n")

        f.write("\n\nUNSUPERVISED MODELS\n")
        f.write("-" * 60 + "\n")
        for model, info in clustering.items():
            f.write(f"  {model}:\n")
            for k, v in info.items():
                f.write(f"    {k}: {v}\n")

        best_name = max(after, key=after.get)
        f.write(f"\n\nBEST OVERALL MODEL: {best_name} — {after[best_name]*100:.2f}%\n")

    print(f"\n  Metrics report saved → {path}")
