from src.data_preprocessing import (
    load_data, preprocess,
    plot_sensor_data, plot_class_distribution,
)
from src.train import (
    split_data,
    train_logistic_regression_default,
    train_svm_default,
    train_knn_default,
    train_kmeans,
    train_dbscan,
    tune_logistic_regression,
    tune_svm,
    tune_knn,
    save_model,
)
from src.evaluate import (
    evaluate_classifier,
    evaluate_clustering,
    compare_models,
    compare_before_after,
    plot_confusion_matrix,
    save_metrics_report,
)


def main():
    # ------------------------------------------------------------------
    # 1. Load raw data & save preprocessing plots
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Loading Data & Saving Preprocessing Plots")
    print("="*60)

    raw_data = load_data()
    print(f"Raw dataset: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")

    print("\nSaving sensor time-series plots...")
    plot_sensor_data(raw_data)

    print("\nSaving raw class distribution pie chart...")
    plot_class_distribution(raw_data["Fault_Detection"], tag="raw")

    # ------------------------------------------------------------------
    # 2. Preprocess & split
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Preprocessing Data")
    print("="*60)

    features, labels, pipeline = preprocess()
    print(f"Dataset size after cleaning: {features.shape[0]} samples")

    print("\nSaving cleaned class distribution pie chart...")
    plot_class_distribution(labels, tag="after_cleaning")

    X_train, X_test, y_train, y_test = split_data(features, labels)
    print(f"Train size : {X_train.shape[0]}")
    print(f"Test  size : {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Train all models with DEFAULT hyperparameters
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Training Models — Default Hyperparameters")
    print("="*60)

    print("\n[Logistic Regression — default: C=0.1, penalty='l2']")
    lr_default = train_logistic_regression_default(X_train, y_train)
    lr_default_acc, _ = evaluate_classifier(lr_default, X_test, y_test, "Logistic Regression (default)")

    print("\n[SVM — default: kernel='linear', C=1.0]")
    svm_default = train_svm_default(X_train, y_train)
    svm_default_acc, _ = evaluate_classifier(svm_default, X_test, y_test, "SVM (default)")

    print("\n[KNN — default: k=3]")
    knn_default = train_knn_default(X_train, y_train)
    knn_default_acc, _ = evaluate_classifier(knn_default, X_test, y_test, "KNN (default, k=3)")

    # ------------------------------------------------------------------
    # 4. Unsupervised models — K-Means & DBSCAN
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Unsupervised Models — K-Means & DBSCAN")
    print("="*60)

    print("\n[K-Means — n_clusters=3]")
    kmeans_model = train_kmeans(X_train)
    kmeans_labels = kmeans_model.predict(X_train)
    kmeans_score = evaluate_clustering(X_train, kmeans_labels, "K-Means")
    save_model(kmeans_model, "kmeans_model.pkl")

    print("\n[DBSCAN — eps=0.5, min_samples=10]")
    dbscan_model = train_dbscan(X_train)
    dbscan_labels = dbscan_model.labels_
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_dbscan = list(dbscan_labels).count(-1)
    dbscan_score = evaluate_clustering(X_train, dbscan_labels, "DBSCAN")

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning with GridSearchCV
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Hyperparameter Tuning (GridSearchCV)")
    print("="*60)

    print("\n[Tuning Logistic Regression — grid: C, penalty, solver]")
    lr_tuned, lr_best_params, lr_val_acc = tune_logistic_regression(X_train, y_train)

    print("\n[Tuning SVM — grid: C, kernel, gamma]")
    svm_tuned, svm_best_params, svm_val_acc = tune_svm(X_train, y_train)

    print("\n[Tuning KNN — grid: n_neighbors, metric]")
    knn_tuned, knn_best_params, knn_val_acc = tune_knn(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate tuned models on test set
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: Evaluating Tuned Models on Test Set")
    print("="*60)

    lr_tuned_acc, _ = evaluate_classifier(lr_tuned, X_test, y_test, "Logistic Regression (tuned)")
    save_model(lr_tuned, "logistic_regression_model.pkl")

    svm_tuned_acc, svm_y_pred = evaluate_classifier(svm_tuned, X_test, y_test, "SVM tuned rbf kernel")
    plot_confusion_matrix(y_test, svm_y_pred, "SVM tuned rbf kernel")
    save_model(svm_tuned, "svm_model.pkl")

    knn_tuned_acc, _ = evaluate_classifier(knn_tuned, X_test, y_test, "KNN (tuned)")
    save_model(knn_tuned, "knn_model.pkl")

    # ------------------------------------------------------------------
    # 7. Compare results & save report
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 7: Results Comparison & Saving Report")
    print("="*60)

    before = {
        "Logistic Regression": lr_default_acc,
        "SVM":                 svm_default_acc,
        "KNN":                 knn_default_acc,
    }
    after = {
        "Logistic Regression": lr_tuned_acc,
        "SVM":                 svm_tuned_acc,
        "KNN":                 knn_tuned_acc,
    }
    compare_before_after(before, after)
    compare_models(after, title="Final Tuned Model Comparison")

    # Save text report
    save_metrics_report(
        before=before,
        after=after,
        clustering={
            "K-Means": {
                "n_clusters": 3,
                "Silhouette Score": round(kmeans_score, 4) if kmeans_score else "N/A",
            },
            "DBSCAN": {
                "eps": 0.5,
                "min_samples": 10,
                "Estimated clusters": n_clusters_dbscan,
                "Noise points": n_noise_dbscan,
                "Silhouette Score": round(dbscan_score, 4) if dbscan_score else "N/A",
            },
        },
        best_params={
            "Logistic Regression": lr_best_params,
            "SVM":                 svm_best_params,
            "KNN":                 knn_best_params,
        },
    )

    print("\nAll outputs saved to results/")


if __name__ == "__main__":
    main()
