import numpy as np
from scipy.stats import ttest_ind

# Deception accuracy values
reasoning = [
    59.00, 75.00, 60.00, 64.67,   # openai
    7.00, 8.00, 32.00, 15.67,     # openai o4-mini
    60.00, 53.00, 69.00,          # claude3.7
    10.00, 42.00, 32.00, 28.00    # deepseekR1
]

non_reasoning = [
    10.00, 12.00, 27.00,          # gpt4.1
    16.33, 43.00, 42.00,          # claude3.5
    42.00, 0.00, 44.00,           # deepseekv3
    1.00, 4.25, 44.67, 16.75      # M+H GPT3.5
]

# Welch's t-test
t_stat, p_value = ttest_ind(reasoning, non_reasoning, equal_var=False)

# Cohen's d for independent samples
def cohens_d_ind(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std

cohen_d = cohens_d_ind(reasoning, non_reasoning)

# Summary statistics
mean_r = np.mean(reasoning)
std_r = np.std(reasoning, ddof=1)

mean_nr = np.mean(non_reasoning)
std_nr = np.std(non_reasoning, ddof=1)

# Output
print("=== Deception Accuracy Comparison ===")
print(f"Reasoning models:     M = {mean_r:.2f}%, SD = {std_r:.2f}")
print(f"Non-reasoning models: M = {mean_nr:.2f}%, SD = {std_nr:.2f}")
print(f"Welch's t = {t_stat:.2f}, p = {p_value:.4f}")
print(f"Cohen’s d = {cohen_d:.2f}")