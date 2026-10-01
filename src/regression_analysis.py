import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

# Set output paths
FIGURES_DIR = os.path.join("report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ---------------------------------------------------------
data_path = os.path.join("data", "garments_worker_productivity.csv")
df = pd.read_csv(data_path)

# Clean string columns
df["department"] = df["department"].str.strip().str.lower()
# Consolidate any typo like 'sweing' -> 'sewing'
df["department"] = df["department"].replace({"sweing": "sewing"})

# Select continuous operational regressors
# Note: 'wip' has ~500 missing values; we exclude it to preserve all 1197 rows
# of production line observations without artificial zero-imputation distortion.
feature_cols = [
    "targeted_productivity",
    "smv",
    "over_time",
    "incentive",
    "idle_time",
    "idle_men",
    "no_of_style_change",
    "no_of_workers",
]
target_col = "actual_productivity"

# Drop rows with NA if any in our chosen feature subset
df_clean = df[feature_cols + [target_col]].dropna().copy()

X_raw = df_clean[feature_cols]
y = df_clean[target_col]

print("=" * 75)
print(f"Dataset successfully loaded. Total observations analyzed: {len(df_clean)}")
print("=" * 75)

# ---------------------------------------------------------
# 2. FIT INITIAL FULL OLS MODEL
# ---------------------------------------------------------
X_full = sm.add_constant(X_raw)
model_full = sm.OLS(y, X_full).fit()

print("\n" + "=" * 75)
print("1. INITIAL FULL REGRESSION MODEL SUMMARY")
print("=" * 75)
print(model_full.summary())

# ---------------------------------------------------------
# 3. BACKWARD ELIMINATION (p-value threshold = 0.05)
# ---------------------------------------------------------
selected_features = list(feature_cols)

while len(selected_features) > 0:
    X_curr = sm.add_constant(df_clean[selected_features])
    fit_curr = sm.OLS(y, X_curr).fit()

    # Exclude constant from p-value check
    p_vals = fit_curr.pvalues.drop("const", errors="ignore")
    max_p = p_vals.max()
    max_var = p_vals.idxmax()

    if max_p > 0.05:
        print(f"Dropping '{max_var}' (p-value = {max_p:.4f} > 0.05)")
        selected_features.remove(max_var)
    else:
        break

# ---------------------------------------------------------
# 4. FIT REFINED / BETTER MODEL
# ---------------------------------------------------------
X_refined = sm.add_constant(df_clean[selected_features])
model_refined = sm.OLS(y, X_refined).fit()

print("\n" + "=" * 75)
print("2. REFINED (BETTER) REGRESSION MODEL SUMMARY")
print("=" * 75)
print(model_refined.summary())

# ---------------------------------------------------------
# 5. MODEL COMPARISON METRICS
# ---------------------------------------------------------
comparison_df = pd.DataFrame(
    {
        "Metric": [
            "Predictors Count",
            "R-squared",
            "Adjusted R-squared",
            "F-statistic",
            "Prob (F-statistic)",
            "AIC",
            "BIC",
        ],
        "Full Model": [
            len(feature_cols),
            round(model_full.rsquared, 4),
            round(model_full.rsquared_adj, 4),
            round(model_full.fvalue, 2),
            f"{model_full.f_pvalue:.4e}",
            round(model_full.aic, 2),
            round(model_full.bic, 2),
        ],
        "Refined Model": [
            len(selected_features),
            round(model_refined.rsquared, 4),
            round(model_refined.rsquared_adj, 4),
            round(model_refined.fvalue, 2),
            f"{model_refined.f_pvalue:.4e}",
            round(model_refined.aic, 2),
            round(model_refined.bic, 2),
        ],
    }
)

print("\n" + "=" * 75)
print("3. MODEL COMPARISON")
print("=" * 75)
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 6. GENERATE DIAGNOSTIC VISUALIZATIONS
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Actual vs Predicted Plot
axes[0].scatter(model_refined.fittedvalues, y, alpha=0.35, edgecolors="k", color="#1f77b4")
axes[0].plot([0, 1.2], [0, 1.2], color="red", linestyle="--", linewidth=1.5, label="Ideal Fit (y = x)")
axes[0].set_title("Actual vs. Predicted Productivity (Refined Model)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Predicted Productivity", fontsize=10)
axes[0].set_ylabel("Actual Productivity", fontsize=10)
axes[0].set_xlim(0.2, 1.1)
axes[0].set_ylim(0.2, 1.2)
axes[0].legend()

# (b) Residuals vs Fitted Plot
residuals = model_refined.resid
axes[1].scatter(model_refined.fittedvalues, residuals, alpha=0.35, edgecolors="k", color="#2ca02c")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Residuals vs. Fitted Values", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Fitted Values", fontsize=10)
axes[1].set_ylabel("Residuals", fontsize=10)

plt.tight_layout()
diagnostic_path = os.path.join(FIGURES_DIR, "model_diagnostics.png")
plt.savefig(diagnostic_path, dpi=300)
plt.close()

# (c) Correlation Matrix Heatmap
plt.figure(figsize=(9, 7))
corr = df_clean[feature_cols + [target_col]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Matrix of Operational Variables", fontsize=12, fontweight="bold")
plt.tight_layout()
corr_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
plt.savefig(corr_path, dpi=300)
plt.close()

print("\n" + "=" * 75)
print(f"Diagnostic plots saved in: {FIGURES_DIR}/")
print("=" * 75)