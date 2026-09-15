"""
EDA then preprocessing for Default of Credit Card Clients (Yeh & Lien, 2009; UCI).

EDA runs on the raw 30,000-row file. Preprocessing then cleans, splits,
encodes, and scales. Encoders/scalers are fit on train only.

    python eda_preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "default+of+credit+card+clients" / "default of credit card clients.xls"
EDA_DIR = BASE_DIR / "eda_outputs"
PROCESSED_DIR = BASE_DIR / "processed"
EDA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.figsize": (11, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

TARGET = "default payment next month"  # 1 = default, 0 = no default
ML_TARGET = "default"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # file has no PAY_1
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CONTINUOUS_COLS = ["LIMIT_BAL", "AGE", *BILL_COLS, *PAYAMT_COLS]
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

TEST_SIZE = 0.20
RANDOM_STATE = 42  # same train/test clients every run

# Codebook labels. EDUCATION 0/5/6 and MARRIAGE 0 are not in Yeh & Lien (2009).
# EDA keeps those labels visible; preprocessing later folds them into Others.
SEX_MAP = {1: "Male", 2: "Female"}
EDU_MAP = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    0: "Undocumented (0)",
    5: "Undocumented (5)",
    6: "Undocumented (6)",
}
MAR_MAP = {1: "Married", 2: "Single", 3: "Others", 0: "Undocumented (0)"}
EDU_ORDER = [EDU_MAP[k] for k in (1, 2, 3, 4, 0, 5, 6)]
MAR_ORDER = [MAR_MAP[k] for k in (1, 2, 3, 0)]
EDU_CLEAN_MAP = {k: EDU_MAP[k] for k in (1, 2, 3, 4)}
MAR_CLEAN_MAP = {k: MAR_MAP[k] for k in (1, 2, 3)}
PAY_MONTH = {
    "PAY_0": "Sep 2005",
    "PAY_2": "Aug 2005",
    "PAY_3": "Jul 2005",
    "PAY_4": "Jun 2005",
    "PAY_5": "May 2005",
    "PAY_6": "Apr 2005",
}


def savefig(name: str) -> None:
    path = EDA_DIR / name
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  saved {path.name}")


def section(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")


def load_data() -> pd.DataFrame:
    """Row 0 is X1..Y codes; row 1 is the real header. header=0 would break dtypes."""
    df = pd.read_excel(DATA_PATH, header=1)
    df.columns = df.columns.str.strip()

    expected_cols = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        *PAY_COLS,
        *BILL_COLS,
        *PAYAMT_COLS,
        TARGET,
    ]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Unexpected columns:\n{list(df.columns)}")
    if df.shape != (30000, 25):
        raise ValueError(f"Unexpected shape {df.shape}; expected (30000, 25)")
    if df["ID"].nunique() != 30000 or df["ID"].min() != 1 or df["ID"].max() != 30000:
        raise ValueError("ID is not a unique 1..30000 key")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values after load")
    return df


# ===========================================================================
# EDA (raw data)
# ===========================================================================
def overview(df: pd.DataFrame) -> None:
    """Print codebook so codes are not mistaken for amounts."""
    section("1. Dataset overview")
    print("Source : Yeh, I.-C., & Lien, C.-H. (2009). Expert Systems with Applications.")
    print("         UCI Default of Credit Card Clients (Taiwan, Apr-Sep 2005).")
    print(f"File   : {DATA_PATH.name}")
    print(f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Target : default payment next month  (1 = default, 0 = no default)")
    print(
        """
Feature groups
  ID            identifier (not a predictor)
  LIMIT_BAL     given credit, NT dollars (individual + family/supplementary)
  SEX           1 = male, 2 = female
  EDUCATION     1 = graduate school, 2 = university, 3 = high school, 4 = others
                0, 5, 6 appear in the file but are not in the original codebook
  MARRIAGE      1 = married, 2 = single, 3 = others; 0 is undocumented
  AGE           years
  PAY_0..PAY_6  repayment status Sep -> Apr 2005 (there is no PAY_1)
                Official codebook: -1 = pay duly; 1..8 = delay of 1..8 months;
                9 = delay of 9+ months (code 9 never appears here).
                Codes 0 and -2 are in the data but not listed by Yeh & Lien.
                Common later interpretation: 0 = revolving credit (min. paid),
                -2 = no consumption. Treat as a convention, not official labels.
  BILL_AMT1..6  bill statement Sep -> Apr 2005 (NT dollars); can be negative
  PAY_AMT1..6   amount paid Sep -> Apr 2005 (NT dollars)
""".rstrip()
    )
    print("Dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


def data_quality(df: pd.DataFrame) -> None:
    """Missing values, undocumented codes, duplicate profiles, negative bills."""
    section("2. Data quality")
    print(f"Missing cells     : {int(df.isna().sum().sum())}")
    print(f"Non-finite cells  : {int((~np.isfinite(df.to_numpy())).sum())}")
    print(f"Duplicate IDs     : {int(df['ID'].duplicated().sum())}")

    feat = df.drop(columns=["ID"])  # same features, different IDs = duplicate profiles
    n_extra = int(feat.duplicated().sum())
    n_in_pairs = int(feat.duplicated(keep=False).sum())
    print(f"Duplicate profiles: {n_extra} extra copies ({n_in_pairs} rows)")
    print("  These are different IDs with identical features+target, mostly")
    print("  inactive (zero bill and zero payment) records. Not ID collisions.")

    print("\nDocumented vs observed category values:")
    specs = [
        ("SEX", {1, 2}),
        ("EDUCATION", {1, 2, 3, 4}),
        ("MARRIAGE", {1, 2, 3}),
        (TARGET, {0, 1}),
    ]
    for col, documented in specs:
        observed = set(df[col].unique())
        extra = sorted(int(v) for v in observed - documented)
        missing = sorted(int(v) for v in documented - observed)
        print(
            f"  {col}: observed={sorted(int(v) for v in observed)} "
            f"extra={extra or '-'} missing_from_data={missing or '-'}"
        )

    print("\nPAY_* observed codes (9 = 9+ months delay is documented but absent):")
    for col in PAY_COLS:
        vals = sorted(int(v) for v in df[col].unique())
        print(f"  {col} ({PAY_MONTH[col]}): {vals}")
    print("  Note: PAY_5 and PAY_6 have no code 1 (one-month delay).")
    print("  Negative PAY values are status codes, not negative amounts.")

    print("\nBILL_AMT negative values (overpayment / credit balance, not load errors):")
    for col in BILL_COLS:
        n_neg = int((df[col] < 0).sum())
        print(f"  {col}: {n_neg:,} ({100 * n_neg / len(df):.2f}%)")

    print("\nPAY_AMT zeros (no payment that month):")
    for col in PAYAMT_COLS:
        n_zero = int((df[col] == 0).sum())
        print(f"  {col}: {n_zero:,} ({100 * n_zero / len(df):.1f}%)")

    print(f"\nLIMIT_BAL range: {df['LIMIT_BAL'].min():,} .. {df['LIMIT_BAL'].max():,} (no zeros)")
    print(f"AGE range      : {df['AGE'].min()} .. {df['AGE'].max()}")


def target_analysis(df: pd.DataFrame) -> None:
    """Class balance. ~22% default → accuracy alone will look high."""
    section("3. Target distribution")
    counts = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
    summary = pd.DataFrame(
        {
            "class": ["No default (0)", "Default (1)"],
            "count": counts.values,
            "percent": pct.round(2).values,
        }
    )
    print(summary.to_string(index=False))
    print(f"Imbalance ratio (majority:minority) = {counts.max() / counts.min():.2f}:1")

    fig, ax = plt.subplots()
    bars = ax.bar(summary["class"], summary["count"], color=["#4C78A8", "#E45756"])
    ax.set_ylabel("Number of clients")
    ax.set_title("Default payment next month")
    for bar, p in zip(bars, summary["percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}\n({p:.1f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, summary["count"].max() * 1.15)
    savefig("01_target_distribution.png")


def descriptive_stats(df: pd.DataFrame) -> None:
    """Skip SEX/EDUCATION/MARRIAGE/PAY_* — those are codes, not interval scales."""
    section("4. Descriptive statistics (continuous only)")
    print("SEX / EDUCATION / MARRIAGE / PAY_* are codes, not interval scales,")
    print("so they are omitted here. See sections 5 and 7 for those distributions.\n")
    numeric = df[CONTINUOUS_COLS]
    stats = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["skew"] = numeric.skew()
    stats["excess_kurtosis"] = numeric.kurtosis()
    print(stats.round(2).to_string())
    stats.round(3).to_csv(EDA_DIR / "descriptive_stats.csv")
    print("\nSaved descriptive_stats.csv")
    print("Skew > 1 and large excess kurtosis on BILL_AMT / PAY_AMT: heavy right tails.")


def _rate_table(labeled: pd.DataFrame, label_col: str, order: list[str]) -> pd.DataFrame:
    """Default rate by category; small_n flags rates based on n < 200."""
    table = (
        labeled.groupby(label_col, observed=False)[TARGET]
        .agg(n="count", n_default="sum", default_rate="mean")
        .reindex(order)
        .dropna(how="all")
    )
    table["n"] = table["n"].astype(int)
    table["n_default"] = table["n_default"].astype(int)
    table["default_rate_pct"] = (table["default_rate"] * 100).round(2)
    table["small_n"] = table["n"] < 200
    return table[["n", "n_default", "default_rate_pct", "small_n"]]


def categorical_analysis(df: pd.DataFrame) -> None:
    """Default rate by SEX / EDUCATION / MARRIAGE. small_n (n<200) is noise."""
    section("5. Demographic categories vs default")
    labeled = df.copy()
    labeled["SEX_label"] = labeled["SEX"].map(SEX_MAP)
    labeled["EDUCATION_label"] = labeled["EDUCATION"].map(EDU_MAP)
    labeled["MARRIAGE_label"] = labeled["MARRIAGE"].map(MAR_MAP)

    tables = {
        "SEX": _rate_table(labeled, "SEX_label", ["Male", "Female"]),
        "EDUCATION": _rate_table(labeled, "EDUCATION_label", EDU_ORDER),
        "MARRIAGE": _rate_table(labeled, "MARRIAGE_label", MAR_ORDER),
    }
    print("small_n = True when n < 200: default rate is noisy, do not over-interpret.\n")
    for name, table in tables.items():
        print(f"{name}:")
        print(table.to_string())
        print()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("SEX_label", ["Male", "Female"], "SEX"),
        ("EDUCATION_label", EDU_ORDER, "EDUCATION"),
        ("MARRIAGE_label", MAR_ORDER, "MARRIAGE"),
    ]
    for ax, (col, order, title) in zip(axes, specs):
        rates = (
            labeled.groupby(col)[TARGET]
            .mean()
            .mul(100)
            .reindex(order)
            .dropna()
            .reset_index()
            .rename(columns={TARGET: "default_rate", col: "category"})
        )
        sns.barplot(data=rates, x="category", y="default_rate", ax=ax, color="#4C78A8")
        ax.set_ylabel("Default rate (%)")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ns = labeled[col].value_counts()
        for patch, category in zip(ax.patches, rates["category"]):
            n = int(ns.get(category, 0))
            ax.annotate(
                f"{patch.get_height():.1f}%\nn={n:,}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(rates["default_rate"].max() * 1.45, 1))
    fig.suptitle("Default rate by demographic category (n on each bar)", y=1.03)
    savefig("02_default_rate_demographics.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, order, title) in zip(axes, specs):
        sns.countplot(
            data=labeled,
            x=col,
            hue=TARGET,
            hue_order=[0, 1],  # legend: No then Yes
            order=order,
            ax=ax,
            palette=["#4C78A8", "#E45756"],
        )
        ax.set_xlabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Default", labels=["No (0)", "Yes (1)"])
    fig.suptitle("Client counts by category and default status", y=1.03)
    savefig("03_counts_demographics.png")


def numeric_distributions(df: pd.DataFrame) -> None:
    """Age means look flat; bins show a U-shape. LIMIT_BAL has an inverse gradient."""
    section("6. Age and credit limit")
    print("AGE by target (means look almost identical):")
    print(df.groupby(TARGET)["AGE"].describe().round(2).to_string())

    age_bin = pd.cut(
        df["AGE"],
        bins=[20, 25, 30, 35, 40, 45, 50, 55, 80],
        labels=["21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-79"],
        right=True,
    )
    age_rates = df.groupby(age_bin, observed=False)[TARGET].agg(n="count", default_rate="mean")
    age_rates["default_rate_pct"] = (age_rates["default_rate"] * 100).round(2)
    print("\nAGE is U-shaped vs default. Means hide this:")
    print(age_rates[["n", "default_rate_pct"]].to_string())

    print("\nLIMIT_BAL by target:")
    print(df.groupby(TARGET)["LIMIT_BAL"].describe().round(2).to_string())
    limit_bin = pd.qcut(df["LIMIT_BAL"], q=5)  # Q1 = lowest credit limit
    limit_rates = df.groupby(limit_bin, observed=False)[TARGET].agg(n="count", default_rate="mean")
    limit_rates["default_rate_pct"] = (limit_rates["default_rate"] * 100).round(2)
    print("\nDefault rate falls as credit limit rises (quintiles):")
    print(limit_rates[["n", "default_rate_pct"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["AGE"], bins=30, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Age distribution")
    sns.histplot(df["LIMIT_BAL"] / 1000, bins=40, kde=True, ax=axes[1], color="#54A24B")
    axes[1].set_title("Credit limit (thousand NTD)")
    axes[1].set_xlabel("LIMIT_BAL (000s)")
    savefig("04_age_limit_hist.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x=TARGET, y="AGE", hue=TARGET, ax=axes[0], palette=["#4C78A8", "#E45756"], legend=False)
    axes[0].set_xticks([0, 1], labels=["No default", "Default"])
    axes[0].set_title("Age by default (similar median)")
    sns.boxplot(data=df, x=TARGET, y="LIMIT_BAL", hue=TARGET, ax=axes[1], palette=["#4C78A8", "#E45756"], legend=False)
    axes[1].set_xticks([0, 1], labels=["No default", "Default"])
    axes[1].set_title("Credit limit by default")
    savefig("05_age_limit_box_by_target.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    age_plot = age_rates.reset_index()
    age_plot["age_bin"] = age_plot.iloc[:, 0].astype(str)
    sns.barplot(data=age_plot, x="age_bin", y="default_rate_pct", ax=axes[0], color="#E45756", order=age_plot["age_bin"])
    axes[0].axhline(100 * df[TARGET].mean(), color="black", linestyle="--", linewidth=1, label="overall")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Default rate (%)")
    axes[0].set_title("Default rate by age (U-shape)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    limit_plot = limit_rates.reset_index()
    limit_plot["limit_bin"] = [f"Q{i+1}" for i in range(len(limit_plot))]
    sns.barplot(data=limit_plot, x="limit_bin", y="default_rate_pct", ax=axes[1], color="#4C78A8")
    axes[1].axhline(100 * df[TARGET].mean(), color="black", linestyle="--", linewidth=1, label="overall")
    axes[1].set_xlabel("LIMIT_BAL quintile (Q1 = lowest)")
    axes[1].set_ylabel("Default rate (%)")
    axes[1].set_title("Default rate by credit-limit quintile")
    axes[1].legend()
    savefig("13_age_limit_default_rates.png")


def repayment_status(df: pd.DataFrame) -> None:
    """Default rate by PAY_* code; pale bars have n < 50."""
    section("7. Repayment status (PAY_0 ... PAY_6)")
    print("Official codebook: -1 = pay duly; 1..8 = months delayed; 9 = 9+ months.")
    print("Observed extras: 0 and -2 (not in Yeh & Lien). Common reading:")
    print("  0 = revolving credit, -2 = no consumption. Not official labels.")
    print("Rates for codes with n < 50 are unstable and flagged below.\n")

    for col in PAY_COLS:
        rates = df.groupby(col)[TARGET].agg(n="count", n_default="sum", default_rate="mean")
        rates["default_rate_pct"] = (rates["default_rate"] * 100).round(1)
        rates["small_n"] = rates["n"] < 50
        print(f"{col} ({PAY_MONTH[col]}):")
        print(rates[["n", "n_default", "default_rate_pct", "small_n"]].to_string())
        print()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, col in zip(axes.ravel(), PAY_COLS):
        rates = df.groupby(col)[TARGET].agg(n="count", default_rate="mean")
        x = rates.index.astype(int)
        y = rates["default_rate"] * 100
        colors = ["#9ECAE9" if n < 50 else "#E45756" for n in rates["n"]]  # pale = unstable n
        ax.bar([str(v) for v in x], y, color=colors)
        ax.set_title(f"{col} ({PAY_MONTH[col]})")
        ax.set_xlabel("Status")
        ax.set_ylabel("Default rate (%)")
        for tick, n, rate in zip(ax.get_xticks(), rates["n"], y):
            ax.text(tick, rate, str(int(n)), ha="center", va="bottom", fontsize=7)
    fig.suptitle("Default rate by repayment status (bar label = n; pale = n<50)")
    savefig("06_default_rate_by_pay_status.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    pay_long = df[PAY_COLS].melt(var_name="month", value_name="status")
    sns.countplot(data=pay_long, x="status", hue="month", ax=ax, palette="viridis")
    ax.set_title("Distribution of repayment status across months")
    ax.set_xlabel("Repayment status code")
    savefig("07_pay_status_counts.png")


def bill_and_payment(df: pd.DataFrame) -> None:
    """Negative bills are overpayments (keep). Inactive and high utilisation default more."""
    section("8. Bill amounts, payments, utilisation")
    amount_cols = BILL_COLS + PAYAMT_COLS
    zero_activity = (df[amount_cols] == 0).all(axis=1)
    print(
        f"Clients with all bills and payments = 0: {int(zero_activity.sum()):,} "
        f"({100 * zero_activity.mean():.2f}%), default rate "
        f"{100 * df.loc[zero_activity, TARGET].mean():.1f}% "
        f"(overall {100 * df[TARGET].mean():.1f}%)"
    )

    any_neg_bill = (df[BILL_COLS] < 0).any(axis=1)
    print(
        f"Clients with any negative bill: {int(any_neg_bill.sum()):,}, "
        f"default rate {100 * df.loc[any_neg_bill, TARGET].mean():.1f}%"
    )

    work = df.copy()
    work["avg_bill"] = work[BILL_COLS].mean(axis=1)  # EDA only; not kept (collinear with BILL_AMT*)
    work["avg_pay"] = work[PAYAMT_COLS].mean(axis=1)
    work["utilization"] = work["BILL_AMT1"] / work["LIMIT_BAL"]  # bill exceeds limit when > 1
    print("\nLatest utilisation BILL_AMT1 / LIMIT_BAL:")
    print(work["utilization"].describe(percentiles=[0.01, 0.5, 0.95, 0.99]).round(3).to_string())
    over = work["utilization"] > 1
    print(
        f"utilisation > 1 (bill exceeds limit): {int(over.sum()):,} "
        f"({100 * over.mean():.1f}%), default rate {100 * work.loc[over, TARGET].mean():.1f}%"
    )
    print(
        f"utilisation <= 1: default rate {100 * work.loc[~over, TARGET].mean():.1f}%"
    )

    print("\nDerived features by target:")
    print(work.groupby(TARGET)[["avg_bill", "avg_pay", "utilization"]].describe().round(2).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), BILL_COLS):
        sns.histplot(np.clip(work[col] / 1000, -50, 400), bins=40, ax=ax, color="#4C78A8")  # clip tails for display
        ax.set_title(col)
        ax.set_xlabel("Amount (000s), clipped to [-50, 400]")
    fig.suptitle("Bill statement amounts (clipped so the bulk of the distribution is visible)")
    savefig("08_bill_amount_hist.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), PAYAMT_COLS):
    sns.histplot(np.log1p(work[col]), bins=40, ax=ax, color="#54A24B")  # log1p: zeros stay 0
        ax.set_title(col)
        ax.set_xlabel("log1p(PAY_AMT)")
    fig.suptitle("Previous payment amounts (log1p; spike at 0 = no payment)")
    savefig("09_pay_amount_log_hist.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = [
        "Average bill amount",
        "Average payment",
        "Latest utilisation (BILL_AMT1 / LIMIT_BAL)",
    ]
    for ax, col, title in zip(axes, ["avg_bill", "avg_pay", "utilization"], titles):
        sns.boxplot(
            data=work,
            x=TARGET,
            y=col,
            hue=TARGET,
            ax=ax,
            palette=["#4C78A8", "#E45756"],
            legend=False,
            showfliers=False,  # hide extreme amounts so the boxes stay readable
        )
        ax.set_xticks([0, 1], labels=["No default", "Default"])
        ax.set_title(title)
    savefig("10_derived_features_by_target.png")


def correlation_analysis(df: pd.DataFrame) -> None:
    """Pearson on amounts + PAY_*; Spearman on PAY_* (ordinal). Skip nominal SEX/EDU/MAR."""
    section("9. Association with default")
    print("Pearson is used only for continuous amounts and for PAY_* as an ordinal")
    print("scale (delay months). SEX / EDUCATION / MARRIAGE are nominal, so they")
    print("are not ranked here; their relationship is the default-rate tables in section 5.")
    print("Spearman is added for PAY_* because the step from 0 -> 1 -> 2 is ordinal")
    print("but not necessarily linear.\n")

    numeric_for_corr = df[CONTINUOUS_COLS + PAY_COLS + [TARGET]]
    pearson = numeric_for_corr.corr(method="pearson")
    target_pearson = pearson[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    print("Pearson r with default:")
    print(target_pearson.round(4).to_string())
    target_pearson.round(4).rename("pearson_r").to_csv(EDA_DIR / "target_correlations.csv")

    spearman_pay = df[PAY_COLS + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    print("\nSpearman rho with default (PAY_* only):")
    print(spearman_pay.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 10))
    plot_df = target_pearson.rename("pearson_r").rename_axis("feature").reset_index()
    sns.barplot(data=plot_df, x="pearson_r", y="feature", ax=ax, color="#4C78A8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("")
    ax.set_title("Correlation with default (continuous + PAY ordinal)")
    savefig("11_target_correlations.png")

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(pearson, dtype=bool))  # one triangle only
    sns.heatmap(
        pearson,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Pearson correlation (amounts, PAY status, target)")
    savefig("12_correlation_heatmap.png")

    high = (
        pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=np.abs, ascending=False)
    )
    print("\nTop 15 pairwise |Pearson r| (excluding self):")
    print(high.head(15).round(3).to_string())


def insights(df: pd.DataFrame) -> None:
    """Verified findings that drive the preprocessing choices below."""
    section("10. Verified EDA findings")
    default_rate = 100 * df[TARGET].mean()
    n0 = int((df[TARGET] == 0).sum())
    n1 = int((df[TARGET] == 1).sum())
    text = f"""
Verified on this file (30,000 rows, ID 1..30000, 0 missing values).

1. Target is imbalanced: {n1:,} defaults ({default_rate:.2f}%) vs {n0:,} non-defaults.
   Majority:minority = {n0/n1:.2f}:1.

2. Strongest signal is recent repayment status. PAY_0 Pearson r = 0.325
   (Spearman 0.292). Default rate is ~13% at codes -2/0, 17% at -1, 34% at 1,
   then jumps to 69% at 2 and 76% at 3. Codes 5-8 are sparse (often n<30);
   do not read a 100% bar as a stable rate (PAY_5 code 8 is one client).

3. PAY_0 is September 2005; there is no PAY_1. PAY_5/PAY_6 never take value 1.
   Codes 0 and -2 are in the data but not in the original Yeh & Lien codebook.

4. Credit limit has a clear inverse gradient: default ~31.8% in the lowest
   LIMIT_BAL quintile vs ~13.8% in the highest. Defaulters also have a lower
   median limit (90,000 vs 150,000).

5. Age means are almost the same (35.4 vs 35.7), but default rate is U-shaped:
   ~26.7% at 21-25 and 26.5% at 56-79, vs ~19.4% at 31-35.

6. Demographics: male 24.2% vs female 20.8%. High school 25.2% vs graduate
   school 19.2%. Married 23.5% vs single 20.9%. EDUCATION 0/5/6 (n=14/280/51)
   and MARRIAGE 0 (n=54) are undocumented; only EDUCATION=5 is large enough
   for a stable rate (~6.4%). Others categories also have small n.

7. BILL_AMT1..6 are highly collinear (r up to 0.95). Average bill is similar
   by class; average payment is lower for defaulters (3,328 vs 5,828).
   2,115 clients (7.0%) have BILL_AMT1 > LIMIT_BAL; their default rate is 30.1%.
   Negative bills (overpay/credit) have a *lower* default rate (16.5%).

8. PAY_AMT is zero in 17-24% of rows depending on month. 795 clients have
   all bills and payments = 0; their default rate is 37.6% (above overall).

9. 35 extra duplicate feature-profiles (70 rows, different IDs), mostly
   inactive accounts. Not missing-value problems and not duplicate IDs.
""".strip()
    print(text)
    (EDA_DIR / "eda_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved {EDA_DIR / 'eda_summary.txt'}")


def run_eda(df: pd.DataFrame) -> None:
    """EDA on the raw file. Does not modify df."""
    overview(df)
    data_quality(df)
    target_analysis(df)
    descriptive_stats(df)
    categorical_analysis(df)
    numeric_distributions(df)
    repayment_status(df)
    bill_and_payment(df)
    correlation_analysis(df)
    insights(df)
    print(f"\nEDA figures and tables saved to: {EDA_DIR}")


# ===========================================================================
# Preprocessing (after EDA)
# ===========================================================================
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Row-wise cleaning. No statistics from other rows are used."""
    report: dict = {}
    work = df.copy()
    report["n_raw"] = int(len(work))

    n_dup_extra = int(work.drop(columns=["ID"]).duplicated().sum())
    # Drop before split so an identical profile cannot land in both train and test.
    work = work.drop_duplicates(subset=[c for c in work.columns if c != "ID"], keep="first")
    report["n_duplicate_extras_dropped"] = n_dup_extra
    report["n_after_dedup"] = int(len(work))

    # Undocumented codebook values -> Others
    edu_undoc = work["EDUCATION"].isin([0, 5, 6])
    mar_undoc = work["MARRIAGE"].eq(0)
    report["education_undocumented_recoded"] = int(edu_undoc.sum())
    report["marriage_undocumented_recoded"] = int(mar_undoc.sum())
    work["EDUCATION"] = work["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    work["MARRIAGE"] = work["MARRIAGE"].replace({0: 3})

    pay_capped = {}
    for col in PAY_COLS:
        n = int((work[col] > 4).sum())
        pay_capped[col] = n
        work[col] = work[col].clip(upper=4)  # rare delay codes 5-8 -> 4+
    report["pay_codes_gt4_capped_to_4"] = pay_capped

    # Row-wise features (no leakage)
    work["utilization"] = work["BILL_AMT1"] / work["LIMIT_BAL"]  # LIMIT_BAL is never 0
    work["avg_pay"] = work[PAYAMT_COLS].mean(axis=1)  # defaulters pay less on average
    work["inactive"] = (work[BILL_COLS + PAYAMT_COLS] == 0).all(axis=1).astype(int)

    work = work.rename(columns={TARGET: ML_TARGET})
    work["SEX"] = work["SEX"].map(SEX_MAP)
    work["EDUCATION"] = work["EDUCATION"].map(EDU_CLEAN_MAP)
    work["MARRIAGE"] = work["MARRIAGE"].map(MAR_CLEAN_MAP)

    if work[CAT_COLS].isna().any().any():
        raise ValueError("Unmapped SEX / EDUCATION / MARRIAGE values")
    if not set(work["EDUCATION"].unique()) <= set(EDU_CLEAN_MAP.values()):
        raise ValueError("EDUCATION recode failed")
    if not set(work["MARRIAGE"].unique()) <= set(MAR_CLEAN_MAP.values()):
        raise ValueError("MARRIAGE recode failed")
    if (work[PAY_COLS] > 4).any().any():
        raise ValueError("PAY cap failed")

    report["n_inactive"] = int(work["inactive"].sum())
    report["n_utilization_gt1"] = int((work["utilization"] > 1).sum())
    report["default_rate_after_clean"] = float(work[ML_TARGET].mean())
    return work, report


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split, same default rate in both sets."""
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[ML_TARGET],  # keep ~22% default in both sets
    )
    if set(train["ID"]) & set(test["ID"]):
        raise ValueError("ID leakage: overlap between train and test")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """One-hot on train only. drop='first' → Male / Graduate school / Married."""
    encoder = OneHotEncoder(
        categories=[
            ["Male", "Female"],
            ["Graduate school", "University", "High school", "Others"],
            ["Married", "Single", "Others"],
        ],
        drop="first",  # Male / Graduate school / Married are the reference levels
        handle_unknown="error",
        sparse_output=False,
    )
    encoder.fit(train[CAT_COLS])
    dummy_names = list(encoder.get_feature_names_out(CAT_COLS))

    def apply(part: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.DataFrame(
            encoder.transform(part[CAT_COLS]),
            columns=dummy_names,
            index=part.index,
        )
        return pd.concat([part.drop(columns=CAT_COLS), dummies], axis=1)

    return apply(train), apply(test), encoder


def scale_numeric(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    """log1p on LIMIT_BAL and PAY_AMT (skewed, >= 0), then z-score. Skip dummies and inactive."""
    log_cols = ["LIMIT_BAL", *PAYAMT_COLS]
    dummy_cols = [
        c
        for c in train.columns
        if c.startswith("SEX_") or c.startswith("EDUCATION_") or c.startswith("MARRIAGE_")
    ]
    skip = {ML_TARGET, "ID", "inactive", *dummy_cols}  # keep 0/1 flags unscaled
    scale_cols = [c for c in train.columns if c not in skip]

    train_s = train.copy()
    test_s = test.copy()
    for col in log_cols:
        train_s[col] = np.log1p(train_s[col])  # compress right skew; values are >= 0
        test_s[col] = np.log1p(test_s[col])

    scaler = StandardScaler()
    scaler.fit(train_s[scale_cols])  # test must use train mean/std
    train_s[scale_cols] = scaler.transform(train_s[scale_cols])
    test_s[scale_cols] = scaler.transform(test_s[scale_cols])
    return train_s, test_s, scaler, scale_cols


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False)
    print(f"  saved {path.name}  shape={df.shape}")


def run_preprocess(raw: pd.DataFrame) -> None:
    section("11. Clean (before split)")
    cleaned, report = clean(raw)
    print(f"Dropped duplicate extras : {report['n_duplicate_extras_dropped']}")
    print(f"Rows after dedup         : {report['n_after_dedup']}")
    print(f"EDUCATION 0/5/6 -> Others: {report['education_undocumented_recoded']}")
    print(f"MARRIAGE 0 -> Others     : {report['marriage_undocumented_recoded']}")
    print("PAY codes > 4 capped to 4:")
    for col, n in report["pay_codes_gt4_capped_to_4"].items():
        print(f"  {col}: {n}")
    print(f"Inactive flag            : {report['n_inactive']}")
    print(f"utilisation > 1          : {report['n_utilization_gt1']}")
    print(f"Default rate after clean : {100 * report['default_rate_after_clean']:.2f}%")
    print("\nCategory counts after recode:")
    for col in CAT_COLS:
        print(f"  {col}: {cleaned[col].value_counts().to_dict()}")
    print("PAY_0 counts after cap:")
    print(f"  {cleaned['PAY_0'].value_counts().sort_index().to_dict()}")

    section("12. Stratified train/test split")
    train, test = split(cleaned)
    print(f"Train: {len(train):,}  default={100 * train[ML_TARGET].mean():.2f}%")
    print(f"Test : {len(test):,}  default={100 * test[ML_TARGET].mean():.2f}%")
    print(f"Test size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify={ML_TARGET}")

    audit = pd.concat(  # keep IDs so we can prove train/test do not overlap
        [
            train[["ID", ML_TARGET]].assign(split="train"),
            test[["ID", ML_TARGET]].assign(split="test"),
        ],
        ignore_index=True,
    )
    audit.to_csv(PROCESSED_DIR / "split_ids.csv", index=False)

    section("13. One-hot encode SEX / EDUCATION / MARRIAGE (fit on train)")
    train_enc, test_enc, encoder = encode_categoricals(train, test)
    dummy_cols = list(encoder.get_feature_names_out(CAT_COLS))
    print(f"Dummy columns (drop first): {dummy_cols}")
    print("Reference levels: SEX=Male, EDUCATION=Graduate school, MARRIAGE=Married")

    train_ml = train_enc.drop(columns=["ID"])  # ID already used to check no overlap
    test_ml = test_enc.drop(columns=["ID"])
    feature_cols = [c for c in train_ml.columns if c != ML_TARGET]
    print(f"Unscaled feature count: {len(feature_cols)}")

    section("14. Scale for ANN / Gaussian NB (fit on train)")
    train_scaled, test_scaled, scaler, scale_cols = scale_numeric(train_ml, test_ml)
    print(f"log1p then StandardScaler on: {scale_cols}")
    print("Not scaled: one-hot dummies, inactive, target")
    print("Train scaled means (should be ~0):")
    print(train_scaled[scale_cols].mean().round(4).to_string())
    print("Train scaled stds (should be ~1):")
    print(train_scaled[scale_cols].std(ddof=0).round(4).to_string())

    section("15. Save processed files")
    save_csv(train_ml, "train.csv")
    save_csv(test_ml, "test.csv")
    save_csv(train_scaled, "train_scaled.csv")
    save_csv(test_scaled, "test_scaled.csv")
    (PROCESSED_DIR / "feature_names.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    joblib.dump(encoder, PROCESSED_DIR / "encoder.joblib")
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")
    print("  saved encoder.joblib, scaler.joblib, feature_names.json, split_ids.csv")

    summary = {
        **report,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int(len(train_ml)),
        "n_test": int(len(test_ml)),
        "train_default_rate": float(train_ml[ML_TARGET].mean()),
        "test_default_rate": float(test_ml[ML_TARGET].mean()),
        "n_features": len(feature_cols),
        "dummy_columns": dummy_cols,
        "scaled_columns": scale_cols,
        "log1p_columns": ["LIMIT_BAL", *PAYAMT_COLS],
        "notes": [
            "Duplicates dropped before split so the same profile cannot sit in both sets.",
            "Negative BILL_AMT values were kept (overpayment / credit).",
            "PAY codes 0 and -2 were kept; values > 4 were capped to 4.",
            "train.csv / test.csv: one-hot, unscaled — use for Decision Tree and Random Forest.",
            "train_scaled.csv / test_scaled.csv: log1p + z-score — use for ANN and Gaussian NB.",
            "Class imbalance is unchanged; handle it at training time (class_weight / resampling).",
        ],
    }
    (PROCESSED_DIR / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    text = f"""
Preprocessing summary
=====================
Raw rows                         : {report['n_raw']:,}
Duplicate extras dropped         : {report['n_duplicate_extras_dropped']}
Rows after clean                 : {report['n_after_dedup']:,}
EDUCATION 0/5/6 -> Others        : {report['education_undocumented_recoded']}
MARRIAGE 0 -> Others             : {report['marriage_undocumented_recoded']}
Train / test                     : {len(train_ml):,} / {len(test_ml):,}
Train default rate               : {100 * train_ml[ML_TARGET].mean():.2f}%
Test default rate                : {100 * test_ml[ML_TARGET].mean():.2f}%
Features (unscaled, one-hot)     : {len(feature_cols)}

Use train.csv + test.csv for Decision Tree and Random Forest.
Use train_scaled.csv + test_scaled.csv for ANN and Gaussian Naive Bayes.
Do not fit scalers or encoders again on the test set.
""".strip()
    (PROCESSED_DIR / "preprocess_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nProcessed files saved to: {PROCESSED_DIR}")


def main() -> None:
    """EDA on raw data, then preprocess. Do not reverse that order."""
    print("Loading dataset...")
    df = load_data()
    print(f"Raw shape: {df.shape}")
    run_eda(df)
    run_preprocess(df)
    section("Done")
    print(f"EDA        -> {EDA_DIR}")
    print(f"Processed  -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
