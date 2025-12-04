"""Visualization helpers for the exercises in the ML folder."""
import argparse
from typing import Callable, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import (load_breast_cancer, load_diabetes, load_iris,
                               make_blobs)
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def visualize_preprocessing() -> None:
    data = np.array([10, 20, 30, 40, 50], dtype=float).reshape(-1, 1)
    minmax_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    normalized = minmax_scaler.fit_transform(data).flatten()
    standardized = std_scaler.fit_transform(data).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(data)), data.flatten(), color="#1f77b4")
    axes[0].set_title("Original Values")
    axes[0].set_xticks(range(len(data)))
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Value")

    axes[1].bar(range(len(normalized)), normalized, color="#ff7f0e", alpha=0.8)
    axes[1].plot(range(len(standardized)), standardized, marker="o", linestyle="--", color="#2ca02c")
    axes[1].set_title("Normalized (bars) vs Standardized (line)")
    axes[1].set_xticks(range(len(data)))
    axes[1].set_xlabel("Index")
    axes[1].legend(["Standardized (z-score)", "Normalized (min-max)"])

    plt.suptitle("Data Preprocessing")
    plt.tight_layout()
    plt.show()


def visualize_linear_regression() -> None:
    X = np.array([1, 3, 5, 2, 3], dtype=float)
    y = np.array([4, 1, 5, 3, 2], dtype=float)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)

    X_plot = np.linspace(X.min() - 1, X.max() + 1, 100)
    y_plot = model.predict(X_plot.reshape(-1, 1))
    residuals = y - model.predict(X.reshape(-1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(X, y, color="#1f77b4", label="Observed")
    axes[0].plot(X_plot, y_plot, color="#ff7f0e", label="Fitted Line")
    axes[0].set_title("Linear Regression Fit")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].stem(range(len(residuals)), residuals, linefmt="#2ca02c", basefmt="k")
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Residual")

    r2_manual = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    fig.suptitle(f"Linear Regression (slope={model.coef_[0]:.2f}, intercept={model.intercept_:.2f}, R^2={r2_manual:.3f})")
    plt.tight_layout()
    plt.show()


def visualize_supervised_classification() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        accuracies[name] = clf.score(X_test_scaled, y_test)

    # Prepare 2D projection for scatter overview
    pca = PCA(n_components=2, random_state=42)
    X_proj = pca.fit_transform(scaler.fit_transform(data.data))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(accuracies.keys(), accuracies.values(), color="#1f77b4")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy on Breast Cancer")
    axes[0].tick_params(axis="x", rotation=20)

    scatter = axes[1].scatter(
        X_proj[:, 0], X_proj[:, 1], c=data.target, cmap="coolwarm", edgecolor="k", alpha=0.7
    )
    axes[1].set_title("PCA Projection of Breast Cancer Data")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    legend1 = axes[1].legend(*scatter.legend_elements(), title="Class")
    axes[1].add_artist(legend1)

    plt.suptitle("Supervised Classification Overview")
    plt.tight_layout()
    plt.show()


def visualize_supervised_regression() -> None:
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
        "SVR": SVR(kernel="rbf"),
    }

    metrics = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        metrics[name] = {
            "r2": r2_score(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(metrics.keys(), [m["r2"] for m in metrics.values()], color="#1f77b4")
    axes[0].set_title("R^2 Scores")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].plot(y_test, label="Actual", color="#000000")
    for name, preds in predictions.items():
        axes[1].plot(preds, label=name)
    axes[1].set_title("Model Predictions vs Actual")
    axes[1].set_ylabel("Disease Progression")
    axes[1].legend()

    plt.suptitle("Supervised Regression Overview")
    plt.tight_layout()
    plt.show()


def _plot_clusters(ax: plt.Axes, data: np.ndarray, labels: Iterable[int], title: str) -> None:
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for color, label in zip(colors, unique_labels):
        mask = np.array(labels) == label
        if label == -1:
            ax.scatter(data[mask, 0], data[mask, 1], marker="x", color="k", label="Noise")
        else:
            ax.scatter(data[mask, 0], data[mask, 1], color=color, label=f"Cluster {label}")
    ax.set_title(title)
    ax.legend(loc="best")


def visualize_unsupervised_clustering() -> None:
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.6, random_state=42)

    kmeans_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    hierarchical_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_clusters(axes[0], X, kmeans_labels, "k-Means")
    _plot_clusters(axes[1], X, hierarchical_labels, "Agglomerative")
    _plot_clusters(axes[2], X, dbscan_labels, "DBSCAN")

    plt.suptitle("Unsupervised Clustering Techniques")
    plt.tight_layout()
    plt.show()


def visualize_dimensionality_reduction() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], color="#1f77b4", alpha=0.7)
    axes[0].set_title(f"PCA (explained variance={pca.explained_variance_ratio_.sum():.2f})")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")

    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], color="#ff7f0e", alpha=0.7)
    axes[1].set_title("t-SNE Embedding")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")

    plt.suptitle("Dimensionality Reduction Techniques")
    plt.tight_layout()
    plt.show()


def main() -> None:
    topic_map: Dict[str, Callable[[], None]] = {
        "preprocessing": visualize_preprocessing,
        "linear_regression": visualize_linear_regression,
        "classification": visualize_supervised_classification,
        "regression": visualize_supervised_regression,
        "clustering": visualize_unsupervised_clustering,
        "dimensionality_reduction": visualize_dimensionality_reduction,
    }

    parser = argparse.ArgumentParser(description="Visualize ML practical outputs.")
    parser.add_argument(
        "--topic",
        choices=list(topic_map.keys()) + ["all"],
        default="all",
        help="Select which visualization to show (default: all sequentially).",
    )
    args = parser.parse_args()

    if args.topic == "all":
        for name, func in topic_map.items():
            print(f"\n--- Showing {name.replace('_', ' ').title()} ---")
            func()
    else:
        topic_map[args.topic]()


if __name__ == "__main__":
    main()
