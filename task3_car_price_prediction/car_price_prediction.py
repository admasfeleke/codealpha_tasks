"""
CodeAlpha Data Science Internship — Task 3: Car Price Prediction with ML
Author: Admasu Feleke Mulatu (CA/DF1/268581)

Predicts used-car selling prices from features (present price, brand
implications via car name, year, mileage, fuel type, transmission, owner
count). Covers data preprocessing, feature engineering, model training
(regression), evaluation, and feature importance analysis.

Dataset: data/car data.csv (301 used cars listed by a dealer)

Usage:
    python car_price_prediction.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render charts to files (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="talk")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "Selling_Price"


def load_data() -> pd.DataFrame:
    """Load the raw CSV and normalize column names."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    renames = {"Driven_kms": "Driven_kms"}
    df = df.rename(columns=renames)
    for col in ("Fuel_Type", "Selling_type", "Transmission"):
        df[col] = df[col].astype(str).str.strip()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: car age, brand, depreciation, per-year usage."""
    df = df.copy()
    max_year = int(df["Year"].max())
    df["Car_Age"] = max_year - df["Year"] + 1
    # Brand (first token of Car_Name) — captures brand goodwill
    df["Brand"] = df["Car_Name"].astype(str).str.strip().str.split().str[0].str.lower()
    # Depreciation so far and annual kilometres
    df["Depreciation"] = df["Present_Price"] - df[TARGET]
    df["Kms_per_Year"] = df["Driven_kms"] / df["Car_Age"].clip(lower=1)
    # Remove extreme price outliers (Toyota Land Cruiser at 92.6 lakhs is
    # 3x the next car — a luxury SUV unrepresentative of the dealer's
    # mainstream inventory). Documented in README.
    n_before = len(df)
    df = df[df["Present_Price"] <= 50].reset_index(drop=True)
    if len(df) < n_before:
        print(f"Removed {n_before - len(df)} extreme outlier row(s) "
              f"(Present_Price > 50 lakhs)")
    # Rare brands grouped into 'other' to limit category explosion
    counts = df["Brand"].value_counts()
    rare = counts[counts < 7].index
    df["Brand"] = df["Brand"].replace(rare, "other")
    return df


def explore(df: pd.DataFrame) -> None:
    """Print EDA stats and save exploratory charts."""
    print("=" * 64)
    print("EXPLORATORY DATA ANALYSIS — Used Car Prices")
    print("=" * 64)
    print(f"Rows: {len(df)} | Features: {df.shape[1] - 1}")
    print(f"\nMissing values per column:\n{df.isna().sum().to_string()}")
    print("\nTarget summary (Selling_Price, lakhs):")
    print(df[TARGET].describe().round(2).to_string())
    print("\nFuel types:", dict(df["Fuel_Type"].value_counts()))
    print("Selling types:", dict(df["Selling_type"].value_counts()))
    print("Transmissions:", dict(df["Transmission"].value_counts()))
    print("Owner counts:", dict(df["Owner"].value_counts()))

    # Distributions of numeric features
    num_cols = ["Selling_Price", "Present_Price", "Driven_kms",
                "Car_Age", "Kms_per_Year"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, num_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric Feature Distributions", fontsize=15)
    fig.savefig(OUT_DIR / "01_feature_distributions.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap (numeric only)
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[num_cols + ["Depreciation"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix (numeric features)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Price vs key categorical features
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col in zip(axes, ["Fuel_Type", "Selling_type", "Transmission"]):
        sns.boxplot(data=df, x=col, y=TARGET, hue=col, legend=False,
                    palette="Set2", ax=ax)
        ax.set_title(f"Selling Price by {col}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_price_by_category.png", dpi=150)
    plt.close(fig)

    # Present vs selling price scatter (strongest relationship)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="Present_Price", y=TARGET,
                    hue="Fuel_Type", ax=ax)
    lims = [0, max(df["Present_Price"].max(), df[TARGET].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1.2, label="y = x (no depreciation)")
    ax.set_title("Present Price vs Selling Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_present_vs_selling.png", dpi=150)
    plt.close(fig)
    print(f"\nEDA charts saved to {OUT_DIR}")
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """One-hot encode categoricals; return X, y and feature names.

    Drops columns that leak or are redundant:
    - Car_Name (captured via Brand), Year (captured via Car_Age)
    - Depreciation and Kms_per_Year are kept only if they don't leak the target
    """
    X = pd.get_dummies(
        df.drop(columns=["Car_Name", "Year", "Depreciation", TARGET]),
        columns=["Fuel_Type", "Selling_type", "Transmission", "Brand"],
        dtype=int)
    y = df[TARGET]
    return X, y, list(X.columns)


def evaluate_models(X: pd.DataFrame, y: pd.Series,
                    feature_names: list) -> pd.DataFrame:
    """Train/test split, train 3 regressors, report metrics + charts."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Ridge Regression": make_pipeline(Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE),
    }

    results = []
    predictions = {}
    print("\n" + "=" * 64)
    print("MODEL TRAINING & EVALUATION (80/20 split)")
    print("=" * 64)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        cv = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        results.append({"Model": name, "R2": round(r2, 4),
                        "MAE (lakhs)": round(mae, 3),
                        "RMSE (lakhs)": round(rmse, 3),
                        "CV MAE (lakhs)": round(-cv.mean(), 3)})
        print(f"\n--- {name} ---")
        print(f"R2: {r2:.4f} | MAE: {mae:.3f} lakhs | RMSE: {rmse:.3f} lakhs")
        print(f"5-fold CV MAE: {-cv.mean():.3f} +/- {cv.std():.3f} lakhs")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    return results_df, predictions, y_test, models


def make_pipeline(est):
    """Scale features then apply the estimator (linear models benefit)."""
    from sklearn.pipeline import make_pipeline as _mp
    return _mp(StandardScaler(), est)


def demo_prediction(df: pd.DataFrame, feature_names: list) -> None:
    """Predict prices for hypothetical cars with the final model."""
    X, y, _ = preprocess(df)
    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(X, y)

    scenarios = [
        # (Present_Price, Driven_kms, Owner, Car_Age, Fuel, Seller, Trans)
        (8.5, 42000, 0, 6, "Petrol", "Dealer", "Manual"),
        (3.2, 85000, 1, 11, "Diesel", "Individual", "Automatic"),
        (15.0, 22000, 0, 3, "Petrol", "Dealer", "Manual"),
    ]
    rows = []
    for present, kms, owner, age, fuel, seller, trans in scenarios:
        row = {name: 0 for name in feature_names}
        row["Present_Price"] = present
        row["Driven_kms"] = kms
        row["Owner"] = owner
        row["Car_Age"] = age
        row["Kms_per_Year"] = kms / max(age, 1)
        fuel_col = f"Fuel_Type_{fuel}"
        seller_col = f"Selling_type_{seller}"
        trans_col = f"Transmission_{trans}"
        if fuel_col in row:
            row[fuel_col] = 1
        if seller_col in row:
            row[seller_col] = 1
        if trans_col in row:
            row[trans_col] = 1
        rows.append(row)
    samples = pd.DataFrame(rows)[feature_names]

    preds = model.predict(samples)
    print("\n" + "=" * 64)
    print("PREDICTION DEMO — hypothetical cars")
    print("=" * 64)
    for (present, kms, owner, age, fuel, seller, trans), p in zip(
            scenarios, preds):
        print(f"{present:>5}L present | {kms:>6} km | {owner} owner(s) | "
              f"{age:>2} yrs | {fuel:<6} | {seller:<10} | {trans:<9} "
              f"-> predicted {p:.2f} lakhs")


def plot_diagnostics(results_df: pd.DataFrame, predictions: dict,
                     y_test: pd.Series, models: dict,
                     feature_names: list) -> None:
    """Predicted-vs-actual, residuals, and feature importance charts."""
    # Predicted vs actual for all models
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, pred) in zip(axes, predictions.items()):
        r2 = r2_score(y_test, pred)
        ax.scatter(y_test, pred, alpha=0.6, edgecolor="k", s=45)
        lims = [0, max(y_test.max(), pred.max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1.5, label="perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Selling Price (lakhs)")
        ax.set_ylabel("Predicted Selling Price (lakhs)")
        ax.set_title(f"{name}\nR2 = {r2:.3f}")
        ax.legend()
    fig.suptitle("Predicted vs Actual Car Prices", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # Residuals for the best model
    best_name = results_df.iloc[0]["Model"]
    best_pred = predictions[best_name]
    residuals = y_test.values - best_pred
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(best_pred, residuals, alpha=0.6, edgecolor="k", s=45)
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel(f"Predicted price — {best_name} (lakhs)")
    ax.set_ylabel("Residual (actual − predicted, lakhs)")
    ax.set_title(f"Residual Plot — {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_residuals.png", dpi=150)
    plt.close(fig)

    # Feature importance from Random Forest
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    top.plot(kind="barh", color="#4c9f70", ax=ax)
    ax.set_title("Random Forest Feature Importance (top 12)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_feature_importance.png", dpi=150)
    plt.close(fig)

    print("\nTop 5 most important features (Random Forest):")
    print(importances.sort_values(ascending=False).head(5).round(3).to_string())


def main() -> None:
    df = engineer_features(load_data())
    explore(df)
    X, y, feature_names = preprocess(df)
    results_df, predictions, y_test, models = evaluate_models(
        X, y, feature_names)
    plot_diagnostics(results_df, predictions, y_test, models, feature_names)
    demo_prediction(df, feature_names)
    print("\nAll charts saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

