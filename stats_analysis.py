import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Define Cohen's d for paired samples
def cohen_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

# Define Rank-Biserial correlation for Wilcoxon signed-rank test
def rank_biserial(x, y):
    diff = x - y
    positive_ranks = np.sum(diff > 0)
    negative_ranks = np.sum(diff < 0)
    n = positive_ranks + negative_ranks
    if n == 0:
        return 0.0
    return (positive_ranks - negative_ranks) / n

# Data for each condition
neutral = np.array([57.5, 94.0, 66.5, 92.5, 94.5, 79.5, 60.0, 99.36])
veracity = np.array([49.5, 95.0, 29.0, 69.0, 93.0, 65.5, 53.5, 97.16])
base_rate = np.array([56.5, 76.5, 39.0, 74.5, 85.0, 55.5, 52.5, 66.67])

# Paired t-tests
ttest_neutral_vs_veracity = ttest_rel(neutral, veracity)
ttest_neutral_vs_base = ttest_rel(neutral, base_rate)
ttest_base_vs_veracity = ttest_rel(base_rate, veracity)

# Cohen's d values
d_neutral_vs_veracity = cohen_d(neutral, veracity)
d_neutral_vs_base = cohen_d(neutral, base_rate)
d_base_vs_veracity = cohen_d(base_rate, veracity)

# Print t-test and Cohen's d results
print("=== Paired t-tests ===")
print(f"Neutral vs Veracity: t={ttest_neutral_vs_veracity.statistic:.4f}, p={ttest_neutral_vs_veracity.pvalue:.4f}, d={d_neutral_vs_veracity:.4f}")
print(f"Neutral vs Base-Rate: t={ttest_neutral_vs_base.statistic:.4f}, p={ttest_neutral_vs_base.pvalue:.4f}, d={d_neutral_vs_base:.4f}")
print(f"Base-Rate vs Veracity: t={ttest_base_vs_veracity.statistic:.4f}, p={ttest_base_vs_veracity.pvalue:.4f}, d={d_base_vs_veracity:.4f}")

# Wilcoxon signed-rank tests
wilcoxon_results = {
    'neutral_vs_veracity': (neutral, veracity),
    'neutral_vs_base_rate': (neutral, base_rate),
    'veracity_vs_base_rate': (veracity, base_rate)
}

wilcoxon_outputs = {}
p_values = []

print("\n=== Wilcoxon Tests ===")
for name, (x, y) in wilcoxon_results.items():
    result = wilcoxon(x, y)
    rb = rank_biserial(x, y)
    wilcoxon_outputs[name] = (result, rb)
    p_values.append(result.pvalue)
    print(f"{name}: statistic={result.statistic:.4f}, p={result.pvalue:.4f}, rank-biserial={rb:.4f}")

# Multiple comparisons correction (Bonferroni)
corrected = multipletests(p_values, method='bonferroni')

print("\n=== Bonferroni Corrected Wilcoxon p-values ===")
for i, name in enumerate(wilcoxon_outputs.keys()):
    print(f"{name}: corrected p={corrected[1][i]:.4f}, significant={corrected[0][i]}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Prepare data for plotting
data = pd.DataFrame({
    'Neutral': neutral,
    'Veracity': veracity,
    'Base-Rate': base_rate
})

# Melt the dataframe for seaborn
data_melted = data.melt(var_name='Condition', value_name='Score')

# Set plot style
sns.set(style="whitegrid")

# Create boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Condition', y='Score', data=data_melted, palette='Set2')
sns.swarmplot(x='Condition', y='Score', data=data_melted, color='black', alpha=0.6)

plt.title("Score Distributions by Condition")
plt.ylabel("Score")
plt.xlabel("Condition")
plt.tight_layout()
plt.show()
