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
    abs_h = abs(cohens_h)
    if abs_h < 0.2:
        effect_size = "Negligible"
    elif abs_h < 0.5:
        effect_size = "Small"
    elif abs_h < 0.8:
        effect_size = "Medium"
    else:
        effect_size = "Large"

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
    apa_results = []

    for r in results_list:
        # Format p-value
        if r['p_value_z'] < 0.001:
            p_formatted = "p < .001"
        elif r['p_value_z'] < 0.01:
            p_formatted = "p < .01"
        elif r['p_value_z'] < 0.05:
            p_formatted = "p < .05"
        else:
            p_formatted = f"p = {r['p_value_z']:.3f}"

        # Format Cohen's h and Z
        h_formatted = f"{r['cohens_h']:.2f}"
        z_formatted = f"z = {r['z_statistic']:.2f}"

        apa_row = {
            "Study": r['study'],
            "Reasoning (%)": f"{r['group1_prop']*100:.1f}",
            "Non-reasoning (%)": f"{r['group2_prop']*100:.1f}",
            "Difference (%)": f"{r['difference']*100:.1f}",
            "95% CI (diff)": f"[{r['z_ci_lower']*100:.1f}, {r['z_ci_upper']*100:.1f}]",
            "Z-test": z_formatted,
            "p-value": p_formatted,
            "Cohen's h": h_formatted,
            "Effect size": r['effect_size']
        }
        apa_results.append(apa_row)

    return pd.DataFrame(apa_results)

# Example usage
if __name__ == "__main__":
    # Study proportions
   # study1_n_oa_o3 = 0.525
   # study1_n_tb_r1 = 0.925
   # study1_n_oa_v3 = 0.54
   # study1_n_tb_v3 = 0.6
   # study1_claude = 0.8334
   # study2_v_oa_o3 = 0.745
   # study2_v_oa_41 = 0.55
   # study2_v_tb_o3 = 0.495
   # study2_v_tb_41 = 0.93
   # study2_v_oa_37 = 0.595
   # study2_v_oa_35 = 0.585
   # study2_v_tb_37 = 0.29
   # study2_v_tb_35 = 0.655
   # study2_v_oa_r1 = 0.61
   # study2_v_oa_v3 = 0.505
   # study2_v_tb_r1 = 0.69
   # study2_v_tb_v3 = 0.535
    study3_br_oa_o3 = 0.67
    study3_br_oa_41 = 0.62
    study3_br_tb_o3 = 0.565
    study3_br_tb_41 = 0.85
    study3_br_oa_37 = 0.63
    study3_br_oa_35 = 0.555
    study3_br_tb_37 = 0.39
    study3_br_tb_35 = 0.555
    study3_br_oa_r1 = 0.565
    study3_br_oa_v3 = 0.525
    study3_br_tb_r1 = 0.745
    study3_br_tb_v3 = 0.525
   # study2_claude = 0.655
   # study2_o3 = 0.495
   # study2_claude = 0.655
   # study2_o3 = 0.495
   # study2_claude = 0.655
   # study3_o3 = 0.565
   # study3_claude = 0.555
    n1 = n2 = 200

    # Run analysis
  #  results1 = analyze_proportion_comparison(study1_n_oa_r1, study1_n_oa_v3, n1, n2, "Study 1")
  #  results2 = analyze_proportion_comparison(study2_v_oa_o3, study2_v_oa_41, n1, n2, "Study 2")
  #  results3 = analyze_proportion_comparison(study2_v_tb_o3, study2_v_tb_41, n1, n2, "Study 2")
  #  results4 = analyze_proportion_comparison(study2_v_oa_37, study2_v_oa_35, n1, n2, "Study 2")
  #  results5 = analyze_proportion_comparison(study2_v_tb_37, study2_v_tb_35, n1, n2, "Study 2")
  #  results6 = analyze_proportion_comparison(study2_v_oa_r1, study2_v_oa_v3, n1, n2, "Study 2")
  #  results7 = analyze_proportion_comparison(study2_v_tb_r1, study2_v_tb_v3, n1, n2, "Study 2")
  #  results2 = analyze_proportion_comparison(study1_n_tb_r1, study1_n_tb_v3, n1, n2, "Study 1")
    results8 = analyze_proportion_comparison(study3_br_oa_o3, study3_br_oa_41, n1, n2, "Study 3")
    results9 = analyze_proportion_comparison(study3_br_tb_o3, study3_br_tb_41, n1, n2, "Study 3")
    results10 = analyze_proportion_comparison(study3_br_oa_37, study3_br_oa_35, n1, n2, "Study 3")
    results11 = analyze_proportion_comparison(study3_br_tb_37, study3_br_tb_35, n1, n2, "Study 3")
    results12 = analyze_proportion_comparison(study3_br_oa_r1, study3_br_oa_v3, n1, n2, "Study 3")
    results13 = analyze_proportion_comparison(study3_br_tb_r1, study3_br_tb_v3, n1, n2, "Study 3")
  #  results3 = analyze_proportion_comparison(study2_o3, study2_claude, n1, n2, "Study 2")
  #  results4 = analyze_proportion_comparison(study3_o3, study3_claude, n1, n2, "Study 3")

    # Print detailed results
    results = [
        analyze_proportion_comparison(study3_br_oa_o3, study3_br_oa_41, n1, n2, "OA O3 vs OA 41"),
        analyze_proportion_comparison(study3_br_tb_o3, study3_br_tb_41, n1, n2, "TB O3 vs TB 41"),
        analyze_proportion_comparison(study3_br_oa_37, study3_br_oa_35, n1, n2, "OA 37 vs OA 35"),
        analyze_proportion_comparison(study3_br_tb_37, study3_br_tb_35, n1, n2, "TB 37 vs TB 35"),
        analyze_proportion_comparison(study3_br_oa_r1, study3_br_oa_v3, n1, n2, "OA R1 vs OA V3"),
        analyze_proportion_comparison(study3_br_tb_r1, study3_br_tb_v3, n1, n2, "TB R1 vs TB V3"),
    ] 

    for r in results:
        print_results(r)

   # print_results(results1)
   # print_results(results2, results3, results4, results5, results6, results7)
   # print_results(results3)

    # APA table
    all_results = [results8, results9, results10, results11, results12, results13]
    apa_table = create_apa_table(all_results)

    print("\n--- APA-Style Results Table ---")
    print(apa_table.to_string(index=False))
