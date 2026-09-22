"""
CodeAlpha Data Science Internship — Task 1: Iris Flower Classification
Author: Admasu Feleke Mulatu (CA/DF1/268581)

Classifies iris flowers (setosa, versicolor, virginica) from their
measurements using Scikit-learn, and evaluates accuracy/performance on
test data.

Usage:
    python iris_classification.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render charts to files (no display needed)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"]


def load_data() -> pd.DataFrame:
    """Load the Iris dataset into a pandas DataFrame with readable names."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    df["species"] = pd.Categorical.from_codes(iris.target, TARGET_NAMES)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print summary statistics and save exploratory charts."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} samples x {df.shape[1] - 1} features")
    print("\nClass balance:")
    print(df["species"].value_counts().to_string())
    print("\nMean measurements per species (cm):")
    print(df.groupby("species", observed=True).mean().round(2).to_string())
    print(f"\nMissing values: {df.isna().sum().sum()}")

    # Feature distributions by species
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, FEATURE_NAMES):
        sns.histplot(data=df, x=col, hue="species", kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Iris Feature Distributions by Species", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_feature_distributions.png", dpi=150)
    plt.close(fig)

    # Boxplots: feature spread per species
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, FEATURE_NAMES):
        sns.boxplot(data=df, x="species", y=col, ax=ax)
        ax.set_title(f"{col} by species")
    fig.suptitle("Feature Spread per Species (Boxplots)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_boxplots_by_species.png", dpi=150)
    plt.close(fig)

    # Pairwise scatter matrix
    pair = sns.pairplot(df, hue="species", corner=True, diag_kind="kde")
    pair.figure.suptitle("Iris Pairwise Feature Relationships", y=1.02, fontsize=14)
    pair.figure.savefig(OUTPUT_DIR / "03_pairplot.png", dpi=150)
    plt.close(pair.figure)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[FEATURE_NAMES].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nEDA charts saved to {OUTPUT_DIR}")
def get_models() -> dict:
    """Return the candidate classifiers, each wrapped with scaling."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "K-Nearest Neighbors (k=5)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF kernel)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)),
    }

def train_and_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Split data, train all models, and report test-set performance."""
    X = df[FEATURE_NAMES]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results = []
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION (80/20 stratified split)")
    print("=" * 60)

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X, y, cv=5)
        results.append({
            "Model": name,
            "Test Accuracy": acc,
            "5-Fold CV Mean": round(cv.mean(), 3),
            "5-Fold CV Std": round(cv.std(), 3),
        })
        print(f"\n--- {name} ---")
        print(f"Test accuracy: {acc:.4f}   "
              f"5-fold CV: {cv.mean():.3f} +/- {cv.std():.3f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        safe = name.split()[0].lower()
        fig.savefig(OUTPUT_DIR / f"05_confusion_{safe}.png", dpi=150)
        plt.close(fig)

    results_df = pd.DataFrame(results).sort_values(
        "Test Accuracy", ascending=False).reset_index(drop=True)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # PCA 2-D projection colored by species
    pca = PCA(n_components=2)
    pts = pca.fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for sp in TARGET_NAMES:
        mask = (y == sp).to_numpy()
        ax.scatter(pts[mask, 0], pts[mask, 1], label=sp, edgecolor="k", s=55)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("PCA Projection of Iris Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_pca_projection.png", dpi=150)
    plt.close(fig)

    return results_df


def demo_prediction(df: pd.DataFrame) -> None:
    """Classify new, unseen flower measurements with retrained models."""
    print("\n" + "=" * 60)
    print("PREDICTION DEMO — classifying new flowers")
    print("=" * 60)
    X, y = df[FEATURE_NAMES], df["species"]
    new_flower = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2],
         [6.0, 2.9, 4.5, 1.5],
         [6.9, 3.1, 5.4, 2.1]],
        columns=FEATURE_NAMES)
    print("New measurements (cm):")
    print(new_flower.to_string(index=False))
    for name, model in get_models().items():
        model.fit(X, y)  # refit on all data for deployment-style use
        preds = model.predict(new_flower)
        print(f"{name}: {list(preds)}")
    print("Expected species in order: setosa, versicolor, virginica")


def main() -> None:
    df = load_data()
    explore_data(df)
    train_and_evaluate(df)
    demo_prediction(df)
    print("\nAll charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

