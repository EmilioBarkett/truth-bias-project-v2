import numpy as np
import scipy.stats as stats
import pandas as pd
from math import sqrt


def analyze_proportion_comparison(p1, p2, n1, n2, study_name=""):
    """
    Analyze the comparison between two proportions with comprehensive statistics.

    Parameters:
    -----------
    p1, p2 : float
        The proportions for group 1 and group 2 (between 0 and 1)
    n1, n2 : int
        The sample sizes for group 1 and group 2
    study_name : str
        Optional name for the study

    Returns:
    --------
    dict
        Dictionary containing all calculated statistics
    """
    # Variance and standard error
    var1 = p1 * (1 - p1) / n1
    var2 = p2 * (1 - p2) / n2
    se_diff = sqrt(var1 + var2)

    # Z-test and confidence interval
    pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)
    pooled_se = sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / pooled_se
    p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    z_critical = 1.96
    z_ci_lower = (p1 - p2) - z_critical * pooled_se
    z_ci_upper = (p1 - p2) + z_critical * pooled_se

    # Cohen's h
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    # Effect size interpretation
    effect_size_thresholds = [(0.2, "Negligible"), (0.5, "Small"), (0.8, "Medium"), (float('inf'), "Large")]
    abs_h = abs(cohens_h)
    effect_size = next((desc for threshold, desc in effect_size_thresholds if abs_h < threshold), "Large")

    # Chi-square test
    obs = np.array([[p1 * n1, (1 - p1) * n1], [p2 * n2, (1 - p2) * n2]])
    chi2, p_chi2, _, _ = stats.chi2_contingency(obs)

    return {
        "study": study_name,
        "group1_prop": p1,
        "group2_prop": p2,
        "difference": p1 - p2,
        "z_statistic": z_stat,
        "p_value_z": p_value_z,
        "z_ci_lower": z_ci_lower,
        "z_ci_upper": z_ci_upper,
        "cohens_h": cohens_h,
        "effect_size": effect_size,
        "chi_square": chi2,
        "p_value_chi2": p_chi2
    }


def print_results(results):
    """
    Print the results in a formatted way
    """
    print(f"\n--- Results for {results['study']} ---")
    print(f"Group 1 proportion: {results['group1_prop']:.3f}")
    print(f"Group 2 proportion: {results['group2_prop']:.3f}")
    print(f"Difference: {results['difference']:.3f}")
    print(f"\nZ-test statistic: {results['z_statistic']:.3f}")
    print(f"Z-test p-value: {results['p_value_z']:.6f}")
    print(f"95% CI for Z-test: [{results['z_ci_lower']:.3f}, {results['z_ci_upper']:.3f}]")
    print(f"\nCohen's h: {results['cohens_h']:.3f} ({results['effect_size']})")
    print(f"\nChi-square statistic: {results['chi_square']:.3f}")
    print(f"Chi-square p-value: {results['p_value_chi2']:.6f}")


def create_apa_table(results_list):
    """
    Create a dataframe with APA-style formatting of results

    Parameters:
    -----------
    results_list : list
        List of result dictionaries

    Returns:
    --------
    pandas.DataFrame
        DataFrame with APA-style results
    """
    def format_p_value(p):
        if p < 0.001:
            return "p < .001"
        elif p < 0.01:
            return "p < .01"
        elif p < 0.05:
            return "p < .05"
        else:
            return f"p = {p:.3f}"

    apa_results = []

    for r in results_list:
        apa_row = {
            "Study": r['study'],
            "Reasoning (%)": f"{r['group1_prop']*100:.1f}",
            "Non-reasoning (%)": f"{r['group2_prop']*100:.1f}",
            "Difference (%)": f"{r['difference']*100:.1f}",
            "95% CI (diff)": f"[{r['z_ci_lower']*100:.1f}, {r['z_ci_upper']*100:.1f}]",
            "Z-test": f"z = {r['z_statistic']:.2f}",
            "p-value": format_p_value(r['p_value_z']),
            "Cohen's h": f"{r['cohens_h']:.2f}",
            "Effect size": r['effect_size']
        }
        apa_results.append(apa_row)

    return pd.DataFrame(apa_results)


def run_comparison_set(data_dict, sample_sizes, comparison_type=None):
    """
    Run a set of comparisons on structured data

    Parameters:
    -----------
    data_dict : dict
        Dictionary with model names as keys and dictionaries of data as values
    sample_sizes : tuple
        Tuple containing (n1, n2) sample sizes
    comparison_type : str, optional
        Type of comparison to include in the study name

    Returns:
    --------
    list
        List of result dictionaries
    """
    n1, n2 = sample_sizes
    results = []
    
    # Organize comparisons
    comparisons = []
    models = list(data_dict.keys())
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            
            for condition in data_dict[model1].keys():
                if condition in data_dict[model2]:
                    p1 = data_dict[model1][condition]
                    p2 = data_dict[model2][condition]
                    
                    study_name = f"{model1} vs {model2}"
                    if comparison_type:
                        study_name = f"{comparison_type}: {study_name} ({condition})"
                    else:
                        study_name = f"{study_name} ({condition})"
                        
                    comparisons.append((p1, p2, study_name))
    
    # Run comparisons
    for p1, p2, study_name in comparisons:
        results.append(analyze_proportion_comparison(p1, p2, n1, n2, study_name))
    
    return results


# Example usage
if __name__ == "__main__":
    # Sample sizes
    n1 = n2 = 200

    # Organize data in a more structured way
    study1_data = {
        "OAI_o3": {"oa": 0.67, "tb": 0.575},
        "OAI_41": {"oa": 0.545, "tb": 0.945},
        "ANT_37": {"oa": 0.695, "tb": 0.665},
        "ANT_35": {"oa": 0.625, "tb": 0.795},
        "DS_r1": {"oa": 0.525, "tb": 0.925},
        "DS_v3": {"oa": 0.54, "tb": 0.6}
    }
    
    study2_data = {
        "OAI_o3": {"oa": 0.745, "tb": 0.575},
        "OAI_41": {"oa": 0.55, "tb": 0.93},
        "ANT_37": {"oa": 0.595, "tb": 0.29},
        "ANT_35": {"oa": 0.625, "tb": 0.795},
        "DS_r1": {"oa": 0.61, "tb": 0.69},
        "DS_v3": {"oa": 0.505, "tb": 0.535}
    }
    
    study3_data = {
        "OAI_o3": {"oa": 0.67, "tb": 0.565},
        "OAI_41": {"oa": 0.62, "tb": 0.85},
        "ANT_37": {"oa": 0.63, "tb": 0.39},
        "ANT_35": {"oa": 0.555, "tb": 0.555},
        "DS_r1": {"oa": 0.565, "tb": 0.745},
        "DS_v3": {"oa": 0.525, "tb": 0.525}
    }

    # Example of specific comparisons from the original code
    specific_comparisons = [
        (study1_data["OAI_o3"]["oa"], study1_data["OAI_41"]["oa"], "OAI o3 vs OAI 41 (oa)"),
        (study1_data["OAI_o3"]["tb"], study1_data["OAI_41"]["tb"], "OAI o3 vs OAI 41 (tb)"),
        (study1_data["ANT_37"]["oa"], study1_data["ANT_35"]["oa"], "ANT 37 vs ANT 35 (oa)"),
        (study1_data["ANT_37"]["tb"], study1_data["ANT_35"]["tb"], "ANT 37 vs ANT 35 (tb)")
    ]
    
    # Run specific comparisons
    print("\n--- Running specific comparisons ---")
    specific_results = []
    for p1, p2, study_name in specific_comparisons:
        result = analyze_proportion_comparison(p1, p2, n1, n2, study_name)
        specific_results.append(result)
        print_results(result)
    
    # Example of automated comparison between models for study 3
    print("\n--- Running automated comparison for Study 3 ---")
    study3_results = run_comparison_set(
        {"OAI_o3": study3_data["OAI_o3"], "OAI_41": study3_data["OAI_41"]},
        (n1, n2),
        "Study 3"
    )
    
    for result in study3_results:
        print_results(result)
    
    # Create APA table
    print("\n--- APA-Style Results Table ---")
    apa_table = create_apa_table(specific_results)
    print(apa_table.to_string(index=False))