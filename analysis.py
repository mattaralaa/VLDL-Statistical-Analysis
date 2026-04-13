# =========================================================
# Author: Alaa Matar
# Project: VLDL Statistical Analysis
# =========================================================
# This project performs an inferential statistical analysis
# to examine whether there is a significant difference in VLDL levels
# between two groups (Gender = 0 and Gender = 1).
# =========================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.rcParams["figure.figsize"] = (8, 5)

print("Libraries loaded successfully.")

# =========================
# 2. LOAD DATA
# =========================
file_name = "DiabetesIII1.csv"

if not os.path.exists(file_name):
    raise FileNotFoundError(f"File '{file_name}' not found in the project folder.")

df = pd.read_csv(file_name)

print("=" * 60)
print("File loaded successfully:", file_name)
print("=" * 60)
print("Dataset shape:", df.shape)

# =========================
# 3. BASIC INFO
# =========================
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# =========================
# 4. DATA CLEANING
# =========================
required_cols = ['Gender', 'VLDL']

missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    raise Exception(f"Missing columns: {missing_required}")

df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
df['VLDL'] = pd.to_numeric(df['VLDL'], errors='coerce')

analysis_df = df.dropna(subset=['Gender', 'VLDL']).copy()
analysis_df = analysis_df[analysis_df['Gender'].isin([0, 1])]

print("\nCleaned data shape:", analysis_df.shape)

# =========================
# 5. SPLIT GROUPS
# =========================
group1 = analysis_df[analysis_df['Gender'] == 1]['VLDL']
group2 = analysis_df[analysis_df['Gender'] == 0]['VLDL']

print("\nGroup sizes:")
print("Group 1:", len(group1))
print("Group 0:", len(group2))

# =========================
# 6. DESCRIPTIVE STATS
# =========================
mean1 = group1.mean()
mean2 = group2.mean()

print("\nDescriptive Statistics:")
print("Group 1 Mean:", round(mean1, 4))
print("Group 0 Mean:", round(mean2, 4))

# =========================
# 7. WELCH TEST
# =========================
t_stat, p_val_t = stats.ttest_ind(group1, group2, equal_var=False)

print("\nWelch Test:")
if p_val_t < 0.001:
    print("p-value: < 0.001")
else:
    print("p-value:", round(p_val_t, 4))

# =========================
# 8. MANN-WHITNEY
# =========================
u_stat, p_val_u = stats.mannwhitneyu(group1, group2)

print("\nMann-Whitney:")
print("p-value:", round(p_val_u, 4))

# =========================
# 9. BOOTSTRAP
# =========================
np.random.seed(42)

boot = []
for _ in range(1000):
    s1 = np.random.choice(group1, len(group1), replace=True)
    s2 = np.random.choice(group2, len(group2), replace=True)
    boot.append(s1.mean() - s2.mean())

boot = np.array(boot)
ci_lower = np.percentile(boot, 2.5)
ci_upper = np.percentile(boot, 97.5)
diff = mean1 - mean2

print("\nBootstrap Results:")
print("Mean Difference:", round(diff, 4))
print("95% CI: [{:.2f}, {:.2f}]".format(ci_lower, ci_upper))

# =========================
# 10. EFFECT SIZE
# =========================
def cohens_d(x, y):
    return (x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)

d = cohens_d(group1, group2)

print("\nEffect Size (Cohen's d):", round(d, 3))

# =========================
# 11. VISUALIZATION
# =========================

# Histogram
plt.hist(group1, alpha=0.5, label="Group 1")
plt.hist(group2, alpha=0.5, label="Group 0")
plt.title("VLDL Distribution")
plt.legend()
plt.show()

# Boxplot
sns.boxplot(x='Gender', y='VLDL', data=analysis_df)
plt.title("VLDL Comparison")
plt.show()

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

if p_val_t < 0.001:
    print("Welch p-value: < 0.001")
else:
    print("Welch p-value:", round(p_val_t, 4))

print("Mann-Whitney p-value:", round(p_val_u, 4))
print("Mean Difference:", round(diff, 4))
print("Confidence Interval: [{:.2f}, {:.2f}]".format(ci_lower, ci_upper))
print("Effect Size (d):", round(d, 3))

print("=" * 60)
print("Analysis completed successfully.")
print("=" * 60)
