import numpy as np
import scipy.stats as stats
import pandas as pd
from math import sqrt
import os


def analyze_proportion_comparison(p1, p2, n1, n2, study_name=""):
    """
    Analyze the comparison between two proportions with comprehensive statistics.
    """

    # Compute counts
    count1 = int(round(p1 * n1))
    count2 = int(round(p2 * n2))
    fail1 = n1 - count1
    fail2 = n2 - count2

    # Z-test and confidence interval
    pooled_p = (count1 + count2) / (n1 + n2)
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

    # Chi-square test with proper counts
    obs = np.array([[count1, fail1], [count2, fail2]])
    chi2, p_chi2, _, _ = stats.chi2_contingency(obs, correction=False)

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
            "Z p-value": format_p_value(r['p_value_z']),
            "Chi²": f"{r['chi_square']:.2f}",
            "Chi² p-value": format_p_value(r['p_value_chi2']),
            "Cohen's h": f"{r['cohens_h']:.2f}",
            "Effect size": r['effect_size']
        }
        apa_results.append(apa_row)

    return pd.DataFrame(apa_results)



def compare_reasoning_non_reasoning_by_firm(csv_path, n1=200, n2=200):
    """
    Compare reasoning vs non-reasoning within each firm and study from CSV data
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    n1, n2 : int
        Sample sizes for both groups
        
    Returns:
    --------
    list
        List of result dictionaries
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return []
        
    # Read the CSV file
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Data loaded successfully. Found {len(df)} rows.")
    
    # Display dataframe info
    print("\nDataframe preview:")
    print(df.head())
    print("\nDataframe columns:", df.columns.tolist())
    
    # Fill empty study values with previous non-empty value
    df['study'] = df['study'].fillna(method='ffill')
    
    # Initialize results list
    results = []
    
    # Get unique studies and firms
    studies = df['study'].unique()
    firms = df['firm'].unique()
    
    print(f"\nFound {len(studies)} studies: {studies}")
    print(f"Found {len(firms)} firms: {firms}")
    
    # For each study and firm
    for study in studies:
        for firm in firms:
            study_firm_df = df[(df['study'] == study) & (df['firm'] == firm)]
            
            print(f"\nAnalyzing {study} - {firm}. Found {len(study_firm_df)} rows.")
            
            # Check if we have both reasoning and non-reasoning data
            if len(study_firm_df) >= 2:
                reasoning_row = study_firm_df[study_firm_df['capability'] == 'reasoning']
                non_reasoning_row = study_firm_df[study_firm_df['capability'] == 'non-reasoning']
                
                print(f"  Reasoning rows: {len(reasoning_row)}")
                print(f"  Non-reasoning rows: {len(non_reasoning_row)}")
                
                # Make sure we have both types
                if len(reasoning_row) > 0 and len(non_reasoning_row) > 0:
                    # Get values for both metrics
                    for metric in ['overall_accuracy', 'truth_bias']:
                        r_value = reasoning_row[metric].values[0]
                        nr_value = non_reasoning_row[metric].values[0]
                        
                        print(f"  Comparing {metric}: {r_value} vs {nr_value}")
                        
                        # Construct study name
                        r_model = reasoning_row['model'].values[0]
                        nr_model = non_reasoning_row['model'].values[0]
                        study_name = f"{study}: {firm} ({r_model} vs {nr_model}) - {metric}"
                        
                        # Analyze comparison
                        result = analyze_proportion_comparison(r_value, nr_value, n1, n2, study_name)
                        results.append(result)
                else:
                    print(f"  Missing either reasoning or non-reasoning data for {study} - {firm}")
            else:
                print(f"  Not enough data for {study} - {firm}")
    
    print(f"\nTotal comparisons made: {len(results)}")
    return results


# Main execution
if __name__ == "__main__":
    # Define sample sizes
    n1 = n2 = 200
    
    print("Starting proportion analysis script...")
    
    # Path to the CSV file - try different possibilities
    possible_paths = [
        "metrics_results - metrics_results (1).csv",
        "metrics_results.csv",
        "metrics_results - metrics_results.csv",
        "metrics_results (1).csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"Found CSV file: {path}")
            break
    
    if csv_path is None:
        print("Could not find CSV file. Please specify the exact path:")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        csv_path = input("Enter CSV file path: ")
    
    try:
        # Run comparisons based on CSV data
        print("\n--- Comparing reasoning vs non-reasoning within each firm ---")
        results = compare_reasoning_non_reasoning_by_firm(csv_path, n1, n2)
        
        if results:
            # Print individual results
            for result in results:
                print_results(result)
            
            # Create and print APA table
            print("\n--- APA-Style Results Table ---")
            apa_table = create_apa_table(results)
            print(apa_table.to_string(index=False))
            
            # Save results to file
            output_file = "proportion_comparison_results-v1.csv"
            apa_table.to_csv(output_file, index=False)
            print(f"\nResults saved to '{output_file}'")
            print(f"Full path: {os.path.join(os.getcwd(), output_file)}")
        else:
            print("No results were generated. Please check the CSV file format.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()