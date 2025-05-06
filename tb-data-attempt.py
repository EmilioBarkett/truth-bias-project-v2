from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import pandas as pd

# Data for each condition
neutral = np.array([57.5, 94.0, 66.5, 92.5, 94.5, 79.5, 60.0, 99.36])
veracity = np.array([49.5, 95.0, 29.0, 69.0, 93.0, 65.5, 53.5, 97.16])
base_rate = np.array([56.5, 76.5, 39.0, 74.5, 85.0, 55.5, 52.5, 66.67])

# Paired t-tests
ttest_neutral_vs_veracity = ttest_rel(neutral, veracity)
ttest_neutral_vs_base = ttest_rel(neutral, base_rate)
ttest_base_vs_veracity = ttest_rel(base_rate, veracity)

# Wilcoxon signed-rank tests
wilcoxon_neutral_vs_veracity = wilcoxon(neutral, veracity)
wilcoxon_neutral_vs_base = wilcoxon(neutral, base_rate)
wilcoxon_base_vs_veracity = wilcoxon(base_rate, veracity)

# Output results
{
    "Paired t-test (Neutral vs Veracity)": ttest_neutral_vs_veracity,
    "Paired t-test (Neutral vs Base-Rate)": ttest_neutral_vs_base,
    "Paired t-test (Base-Rate vs Veracity)": ttest_base_vs_veracity,
    "Wilcoxon test (Neutral vs Veracity)": wilcoxon_neutral_vs_veracity,
    "Wilcoxon test (Neutral vs Base-Rate)": wilcoxon_neutral_vs_base,
    "Wilcoxon test (Base-Rate vs Veracity)": wilcoxon_base_vs_veracity
}

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Your scores
neutral_scores = [57.5, 94.0, 66.5, 92.5, 94.5, 79.5, 60.0, 99.36]
veracity_scores = [49.5, 95.0, 29.0, 69.0, 93.0, 65.5, 53.5, 97.16]
base_rate_scores = [56.5, 76.5, 39.0, 74.5, 85.0, 55.5, 52.5, 66.67]

# Perform Wilcoxon signed-rank tests
comparisons = {
    'neutral_vs_veracity': wilcoxon(neutral_scores, veracity_scores),
    'neutral_vs_base_rate': wilcoxon(neutral_scores, base_rate_scores),
    'veracity_vs_base_rate': wilcoxon(veracity_scores, base_rate_scores)
}

# Multiple comparisons correction (Bonferroni)
p_values = [result.pvalue for result in comparisons.values()]
corrected = multipletests(p_values, method='bonferroni')

# Print results
for i, (name, result) in enumerate(comparisons.items()):
    print(f"{name}: statistic={result.statistic:.4f}, p={result.pvalue:.4f}, corrected p={corrected[1][i]:.4f}, significant={corrected[0][i]}")