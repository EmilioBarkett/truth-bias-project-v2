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
    # Calculate variances
    var1 = p1 * (1-p1) / n1
    var2 = p2 * (1-p2) / n2
    
    # Calculate standard errors
    se1 = sqrt(var1)
    se2 = sqrt(var2)
    se_diff = sqrt(var1 + var2)
    
    # Calculate pooled proportion
    pooled_p = (n1*p1 + n2*p2) / (n1 + n2)
    pooled_se = sqrt(pooled_p * (1-pooled_p) * (1/n1 + 1/n2))
    
    # Calculate z-test (standard test for proportions)
    z_stat = (p1 - p2) / se_diff
    p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Calculate Welch's t-test (accounts for unequal variances)
    t_stat = (p1 - p2) / sqrt(var1 + var2)
    # Welch-Satterthwaite degrees of freedom
    df = ((var1 + var2)**2) / ((var1**2 / (n1-1)) + (var2**2 / (n2-1)))
    p_value_t = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Calculate Cohen's d
    # For proportions, we use the pooled standard deviation
    pooled_sd = sqrt(pooled_p * (1-pooled_p))
    cohens_d = (p1 - p2) / pooled_sd
    
    # Calculate confidence intervals
    ci_z_95 = 1.96 * se_diff
    lower_ci = p1 - p2 - ci_z_95
    upper_ci = p1 - p2 + ci_z_95
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "Small"
    elif abs(cohens_d) < 0.8:
        effect_size = "Medium"
    else:
        effect_size = "Large"
    
    # Chi-square test (alternative approach)
    obs = np.array([[p1*n1, (1-p1)*n1], [p2*n2, (1-p2)*n2]])
    chi2, p_chi2, _, _ = stats.chi2_contingency(obs)
    
    # Create results dictionary
    results = {
        "study": study_name,
        "group1_prop": p1,
        "group2_prop": p2,
        "difference": p1 - p2,
        "z_statistic": z_stat,
        "p_value_z": p_value_z,
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value_t": p_value_t,
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "95_ci_lower": lower_ci,
        "95_ci_upper": upper_ci,
        "chi_square": chi2,
        "p_value_chi2": p_chi2
    }
    
    return results

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
    print(f"\nWelch's t-test statistic: {results['t_statistic']:.3f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']:.1f}")
    print(f"Welch's t-test p-value: {results['p_value_t']:.6f}")
    print(f"\nCohen's d: {results['cohens_d']:.3f} ({results['effect_size']})")
    print(f"95% CI: [{results['95_ci_lower']:.3f}, {results['95_ci_upper']:.3f}]")
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
        # Format p-value with APA style
        if r['p_value_t'] < 0.001:
            p_formatted = "p < .001"
        elif r['p_value_t'] < 0.01:
            p_formatted = "p < .01"
        elif r['p_value_t'] < 0.05:
            p_formatted = "p < .05"
        else:
            p_formatted = f"p = {r['p_value_t']:.3f}"
            
        # Format Cohen's d with sign
        d_formatted = f"{r['cohens_d']:.2f}"
        
        # Format t-statistic with degrees of freedom
        t_formatted = f"t({r['degrees_of_freedom']:.0f}) = {r['t_statistic']:.2f}"
        
        apa_row = {
            "Study": r['study'],
            "Reasoning (%)": f"{r['group1_prop']*100:.1f}",
            "Non-reasoning (%)": f"{r['group2_prop']*100:.1f}",
            "Difference (%)": f"{r['difference']*100:.1f}",
            "Welch's t-test": t_formatted,
            "p-value": p_formatted,
            "Cohen's d": d_formatted,
            "Effect size": r['effect_size']
        }
        apa_results.append(apa_row)
    
    return pd.DataFrame(apa_results)

# Example usage with your data
if __name__ == "__main__":
    # Your data
    study1_o3 = 0.575
    study1_claude = 0.795
    n1 = n2 = 200
    
    study2_o3 = 0.495
    study2_claude = 0.655
    
    study3_o3 = 0.565
    study3_claude = 0.555
    
    # Analyze all studies
    results1 = analyze_proportion_comparison(study1_o3, study1_claude, n1, n2, "Study 1")
    results2 = analyze_proportion_comparison(study2_o3, study2_claude, n1, n2, "Study 2")
    results3 = analyze_proportion_comparison(study3_o3, study3_claude, n1, n2, "Study 3")
    
    # Print detailed results
    print_results(results1)
    print_results(results2)
    print_results(results3)
    
    # Create APA-style table
    all_results = [results1, results2, results3]
    apa_table = create_apa_table(all_results)
    
    print("\n--- APA-Style Results Table ---")
    print(apa_table.to_string(index=False))
    
    # You can also export to CSV for use in a spreadsheet program
    # apa_table.to_csv("model_comparison_results.csv", index=False)