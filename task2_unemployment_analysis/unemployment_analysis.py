"""
CodeAlpha Data Science Internship — Task 2: Unemployment Analysis with Python
Author: Admasu Feleke Mulatu (CA/DF1/268581)

Analyzes unemployment rate data in India (2019-2020): data cleaning,
exploration and visualization of unemployment trends, the impact of
Covid-19 (lockdown) on unemployment, and key patterns/seasonal trends.

Datasets (CodeAlpha-provided, from Kaggle):
    data/Unemployment in India.csv           — monthly, states, Rural/Urban
    data/Unemployment_Rate_upto_11_2020.csv  — monthly Jan-Nov 2020, zones, geo

Usage:
    python unemployment_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render charts to files (no display needed)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.autolayout"] = True

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RATE = "Unemployment Rate (%)"
EMPLOYED = "Employed"
LFPR = "Labour Participation Rate (%)"

LOCKDOWN_START = pd.Timestamp("2020-03-25")  # India nationwide lockdown


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, types, and drop broken rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Estimated Unemployment Rate (%)": RATE,
        "Estimated Employed": EMPLOYED,
        "Estimated Labour Participation Rate (%)": LFPR,
        "Region.1": "Zone",
    })
    # The second file has a duplicated Region column (zone names)
    if "Region" in df.columns and "Zone" not in df.columns \
            and df["Region"].dtype == object and df["Region"].nunique() <= 6:
        first_vals = set(df["Region"].unique())
        if first_vals <= {"East", "West", "North", "South", "Northeast"}:
            df = df.rename(columns={"Region": "Zone"})
    df["Region"] = df["Region"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(),
                                dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", RATE])
    for col in (RATE, EMPLOYED, LFPR):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[RATE])
    if "Area" in df.columns:
        df["Area"] = df["Area"].astype(str).str.strip()
    if "Frequency" in df.columns:
        df["Frequency"] = df["Frequency"].astype(str).str.strip()
    df = df.drop_duplicates()
    return df.sort_values("Date").reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both datasets."""
    long_df = _clean_frame(pd.read_csv(DATA_DIR / "Unemployment in India.csv"))
    covid_df = _clean_frame(
        pd.read_csv(DATA_DIR / "Unemployment_Rate_upto_11_2020.csv"))
    return long_df, covid_df

def explore(df: pd.DataFrame, covid_df: pd.DataFrame) -> None:
    """Print summary statistics and dataset overview."""
    print("=" * 64)
    print("EXPLORATORY DATA ANALYSIS — Unemployment in India")
    print("=" * 64)
    print(f"Rows: {len(df)} | States/Regions: {df['Region'].nunique()} | "
          f"Period: {df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}")
    if "Area" in df.columns:
        print(f"Areas: {sorted(df['Area'].unique())}")
    print(f"\nMissing values per column:\n{df.isna().sum().to_string()}")

    print("\nUnemployment rate summary (%):")
    print(df[RATE].describe().round(2).to_string())

    print("\nAverage unemployment rate by Area (%):")
    print(df.groupby("Area")[RATE].mean().round(2).to_string())

    print("\nTop 5 states by average unemployment rate (%):")
    print(df.groupby("Region")[RATE].mean().sort_values(ascending=False)
          .head(5).round(2).to_string())

    print("\nTop 5 states by PEAK unemployment rate (%):")
    print(df.groupby("Region")[RATE].max().sort_values(ascending=False)
          .head(5).round(2).to_string())

    zones = sorted(covid_df["Zone"].unique()) if "Zone" in covid_df.columns else "n/a"
    print(f"\nCovid dataset rows: {len(covid_df)} | "
          f"Period: {covid_df['Date'].min():%b %Y} to {covid_df['Date'].max():%b %Y} | "
          f"Zones: {zones}")


def plot_trends(df: pd.DataFrame) -> None:
    """National monthly trend + area comparison + state heatmap + boxplots."""
    # 1. National monthly average unemployment rate over time
    national = df.groupby("Date")[RATE].mean()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(national.index, national.values, color="crimson", lw=2.2,
            marker="o", ms=4)
    ax.axvspan(pd.Timestamp("2020-03-25"), df["Date"].max(),
               color="crimson", alpha=0.12, label="Covid-19 lockdown period")
    ax.set_title("Average Unemployment Rate in India Over Time (2019-2020)")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    fig.savefig(OUT_DIR / "01_national_trend.png", dpi=150)
    plt.close(fig)

    # 2. Rural vs Urban monthly trends
    if "Area" in df.columns:
        area_trend = df.groupby(["Date", "Area"])[RATE].mean().reset_index()
        fig, ax = plt.subplots(figsize=(13, 5.5))
        sns.lineplot(data=area_trend, x="Date", y=RATE, hue="Area",
                     marker="o", ax=ax)
        ax.axvspan(pd.Timestamp("2020-03-25"), df["Date"].max(),
                   color="crimson", alpha=0.12)
        ax.set_title("Unemployment Rate: Rural vs Urban India")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.savefig(OUT_DIR / "02_rural_vs_urban.png", dpi=150)
        plt.close(fig)

    # 3. State-wise heatmap (state x month)
    heat = df.assign(Month=df["Date"].dt.strftime("%b %y")).pivot_table(
        index="Region", columns="Month", values=RATE, aggfunc="mean")
    order = sorted(heat.columns, key=lambda m: pd.to_datetime(m, format="%b %y"))
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(heat[order], cmap="YlOrRd", linewidths=0.4, ax=ax,
                cbar_kws={"label": RATE})
    ax.set_title("State-wise Unemployment Rate Heatmap (2019-2020)")
    fig.savefig(OUT_DIR / "03_state_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Distribution by Area
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x="Area", y=RATE, hue="Area", legend=False,
                palette="Set2", ax=ax)
    ax.set_title("Unemployment Rate Distribution: Rural vs Urban")
    fig.savefig(OUT_DIR / "04_area_boxplot.png", dpi=150)
    plt.close(fig)


def analyze_covid_impact(df: pd.DataFrame, covid_df: pd.DataFrame) -> dict:
    """Quantify the Covid-19 lockdown impact and produce charts."""
    print("\n" + "=" * 64)
    print("COVID-19 IMPACT ANALYSIS")
    print("=" * 64)
    national = df.groupby("Date")[[RATE, LFPR]].mean()
    pre = national.loc[national.index < LOCKDOWN_START, RATE]
    during = national.loc[national.index >= LOCKDOWN_START, RATE]

    print(f"Pre-lockdown average (May 2019 - Mar 2020): {pre.mean():.2f}%")
    print(f"Lockdown-period average (Apr 2020 - Nov 2020): {during.mean():.2f}%")
    print(f"Absolute rise: {during.mean() - pre.mean():+.2f} percentage points")
    print(f"Relative rise: {(during.mean() / pre.mean() - 1) * 100:+.1f}%")
    peak_day = national[RATE].idxmax()
    print(f"National monthly peak: {national[RATE].max():.2f}% on {peak_day:%B %Y}")

    # Labour participation drop = people leaving the workforce entirely
    pre_lfpr = national.loc[national.index < LOCKDOWN_START, LFPR].mean()
    during_lfpr = national.loc[national.index >= LOCKDOWN_START, LFPR].mean()
    print(f"Labour Participation Rate: {pre_lfpr:.2f}% -> {during_lfpr:.2f}% "
          f"({during_lfpr - pre_lfpr:+.2f} pp) — workers exiting the labour force")

    # Employed-persons collapse during the April 2020 shock
    covid_monthly = covid_df.groupby("Date").agg(
        rate=(RATE, "mean"), employed=(EMPLOYED, "sum"))
    if len(covid_monthly) > 1:
        apr = covid_monthly[
            covid_monthly.index.strftime("%Y-%m") == "2020-04"]
        feb = covid_monthly[
            covid_monthly.index.strftime("%Y-%m") == "2020-02"]
        if len(apr) and len(feb):
            drop = (1 - apr["employed"].iloc[0] / feb["employed"].iloc[0]) * 100
            print(f"Estimated Employed (28 states): Feb 2020 -> Apr 2020: {drop:.1f}% drop")

    # Chart: 2020 zoom with employed bars
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(covid_monthly.index, covid_monthly["rate"], color="crimson",
             lw=2.5, marker="o", label="Unemployment rate")
    ax1.axvline(LOCKDOWN_START, color="k", ls="--", lw=1.5,
                label="Lockdown start (25 Mar 2020)")
    ax1.set_ylabel("Unemployment Rate (%)", color="crimson")
    ax1.set_title("Covid-19 Lockdown Impact on Unemployment and Employment (2020)")
    ax2 = ax1.twinx()
    ax2.bar(covid_monthly.index, covid_monthly["employed"] / 1e6, alpha=0.25,
            color="steelblue", width=18, label="Estimated employed")
    ax2.set_ylabel("Estimated Employed (millions)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11)
    fig.savefig(OUT_DIR / "05_covid_impact_2020.png", dpi=150)
    plt.close(fig)

    # Chart: before/after by zone
    if "Zone" in covid_df.columns:
        zone_cmp = covid_df.assign(
            Phase=lambda d: d["Date"].map(
                lambda t: "Pre-lockdown (Jan-Mar 2020)"
                if t < LOCKDOWN_START else "Lockdown (Apr-Nov 2020)")
        ).groupby(["Zone", "Phase"])[RATE].mean().reset_index()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.barplot(data=zone_cmp, x="Zone", y=RATE, hue="Phase",
                    palette=["#4c9f70", "#d1495b"], ax=ax)
        ax.set_title("Average Unemployment Rate by Zone: Before vs During Lockdown")
        ax.set_ylabel("Unemployment Rate (%)")
        fig.savefig(OUT_DIR / "06_zone_before_after.png", dpi=150)
        plt.close(fig)

    return {"pre": pre.mean(), "during": during.mean(),
            "peak": national[RATE].max(), "peak_date": peak_day}


def seasonal_pattern(df: pd.DataFrame) -> None:
    """Check for seasonal patterns in unemployment across the year."""
    print("\n" + "=" * 64)
    print("SEASONAL / PATTERN ANALYSIS")
    print("=" * 64)
    df = df.assign(Month=df["Date"].dt.month_name())
    cal = df.groupby("Month")[RATE].mean()
    order = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    cal = cal.reindex([m for m in order if m in cal.index])
    print("Average unemployment rate by calendar month (%):")
    print(cal.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(x=cal.index, y=cal.values, hue=cal.index, legend=False,
                palette="flare", ax=ax)
    ax.set_title("Seasonal Pattern: Average Unemployment Rate by Month")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(OUT_DIR / "07_seasonal_pattern.png", dpi=150)
    plt.close(fig)


def state_ranking(df: pd.DataFrame) -> None:
    """Rank states by average unemployment; chart the extremes."""
    means = df.groupby("Region")[RATE].mean().sort_values(ascending=False)
    print("\n" + "=" * 64)
    print("STATE RANKING (average unemployment rate, %)")
    print("=" * 64)
    print("Highest 5:\n" + means.head(5).round(2).to_string())
    print("Lowest 5:\n" + means.tail(5).round(2).to_string())

    extremes = pd.concat([means.head(10), means.tail(10)])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d1495b" if v > means.median() else "#4c9f70" for v in extremes]
    ax.barh(extremes.index, extremes.values,
            color=["#d1495b" if v > means.median() else "#4c9f70"
                   for v in extremes.values])
    ax.invert_yaxis()
    ax.set_title("Top 10 Highest vs Lowest Average Unemployment States")
    ax.set_xlabel("Average Unemployment Rate (%)")
    fig.savefig(OUT_DIR / "08_state_extremes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    long_df, covid_df = load_data()
    explore(long_df, covid_df)
    plot_trends(long_df)
    seasonal_pattern(long_df)
    state_ranking(long_df)

    print("\n" + "=" * 64)
    print("KEY INSIGHTS")
    print("=" * 64)
    stats = analyze_covid_impact(long_df, covid_df)
    print(f"""
1. Covid-19 lockdown (25 Mar 2020) caused an unprecedented spike in
   unemployment: national average jumped from {stats['pre']:.1f}% (pre-lockdown)
   to {stats['during']:.1f}% (lockdown period), peaking at {stats['peak']:.1f}% in {stats['peak_date']:%B %Y}.
2. The shock was sharpest in urban areas — city lockdowns closed services
   and informal work, and urban unemployment overtook rural during 2020.
3. Unemployment is highly uneven across states (e.g. Haryana/Tripura vs
   Meghalaya etc.) — policy needs state-level targeting, not one-size-fits-all.
4. The drop in Labour Participation Rate shows many Indians stopped seeking
   work entirely during the lockdown — hidden unemployment beyond the rate.
5. Policy implications: strengthen urban informal-sector safety nets,
   pandemic-responsive employment schemes, and state-specific interventions.
""")
    print(f"All charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
