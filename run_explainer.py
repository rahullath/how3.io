#!/usr/bin/env python
"""
Run the Project Score Explainer tool with a simple interface.
This script helps you explore how crypto project scores are calculated.
"""
import os
import sys
import glob
from project_score_explainer import ProjectScoreExplainer

def find_latest_file(patterns):
    """Find the most recently modified file matching any of the patterns."""
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # Return the most recently modified file
    return max(all_files, key=os.path.getmtime)

def main():
    print("\n" + "="*50)
    print("CRYPTO PROJECT SCORE EXPLAINER")
    print("="*50)
    
    # Find data files
    data_patterns = ["*.xlsx", "*.csv", "*master*.xlsx", "*master*.csv"]
    results_patterns = ["*result*.csv", "crypto_*.csv"]
    
    data_file = find_latest_file(data_patterns)
    results_file = find_latest_file(results_patterns)
    
    # Get data file
    if data_file:
        print(f"\nFound data file: {data_file}")
        use_found_data = input("Use this file? (Y/n): ").strip().lower()
        if use_found_data == 'n':
            data_file = input("Enter path to data file: ").strip()
    else:
        print("\nNo data file found automatically.")
        data_file = input("Enter path to data file: ").strip()
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)
    
    # Get results file
    if results_file:
        print(f"\nFound results file: {results_file}")
        use_found_results = input("Use this file? (Y/n): ").strip().lower()
        if use_found_results == 'n':
            results_file = input("Enter path to results file (or leave blank): ").strip()
            if not results_file:
                results_file = None
    else:
        print("\nNo results file found automatically.")
        results_file = input("Enter path to results file (or leave blank): ").strip()
        if not results_file:
            results_file = None
    
    if results_file and not os.path.exists(results_file):
        print(f"Warning: Results file '{results_file}' not found. Continuing without it.")
        results_file = None
    
    # Initialize explainer
    explainer = ProjectScoreExplainer(data_file, results_file)
    
    # List all projects
    all_projects = []
    if explainer.df is not None and 'Project' in explainer.df.columns:
        all_projects = explainer.df['Project'].dropna().unique().tolist()
        print(f"\nFound {len(all_projects)} projects in the dataset.")
    
    # Get project name
    while True:
        project_name = input("\nEnter project name to analyze: ").strip()
        
        # Check if project exists
        if project_name:
            if project_name.lower() in [p.lower() for p in all_projects]:
                break
            else:
                print("Project not found. Here are some options:")
                # Show a few examples
                examples = all_projects[:5] if len(all_projects) > 5 else all_projects
                for example in examples:
                    print(f"- {example}")
                continue_anyway = input("Continue anyway? (y/N): ").strip().lower()
                if continue_anyway == 'y':
                    break
        else:
            # If empty, show some examples
            print("Please enter a project name. Some examples:")
            examples = all_projects[:5] if len(all_projects) > 5 else all_projects
            for example in examples:
                print(f"- {example}")
    
    # Get output preferences
    print("\nOutput Options:")
    
    # Visualization
    create_viz = input("Generate visualization? (Y/n): ").strip().lower() != 'n'
    viz_path = None
    if create_viz:
        default_viz_path = f"{project_name}_score_analysis.png"
        viz_path_input = input(f"Visualization file path ({default_viz_path}): ").strip()
        viz_path = viz_path_input if viz_path_input else default_viz_path
    
    # HTML explanation
    create_html = input("Generate HTML explanation? (Y/n): ").strip().lower() != 'n'
    html_path = None
    if create_html:
        default_html_path = f"{project_name}_score_explanation.html"
        html_path_input = input(f"HTML file path ({default_html_path}): ").strip()
        html_path = html_path_input if html_path_input else default_html_path
    
    # Verbose output
    verbose = input("Show detailed calculation steps? (Y/n): ").strip().lower() != 'n'
    
    print("\n" + "="*50)
    print(f"ANALYZING PROJECT: {project_name}")
    print("="*50 + "\n")
    
    # Generate explanation
    results = explainer.explain_project_score(project_name, verbose)
    
    if not results:
        print("\nFailed to generate results for this project.")
        sys.exit(1)
    
    # Create outputs
    if create_viz and viz_path:
        print(f"\nGenerating visualization to: {viz_path}")
        explainer.visualize_project_score(results, viz_path)
    
    if create_html and html_path:
        print(f"\nGenerating HTML explanation to: {html_path}")
        explainer.export_explanation_to_html(results, html_path)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    # If we created HTML, offer to open it
    if create_html and html_path and os.path.exists(html_path):
        open_html = input("\nOpen HTML explanation in browser? (Y/n): ").strip().lower() != 'n'
        if open_html:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(html_path)}")

if __name__ == "__main__":
    main()