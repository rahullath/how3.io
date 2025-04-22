import pandas as pd
import json
import os

def add_json_column_to_csv(csv_file, json_file, output_csv_file, json_column_name='UserGrowthJSON'):
    """
    Add a new column to the CSV containing JSON data for each project from the specified JSON file,
    and transform the CSV to match the Google Sheet format.
    
    Args:
        csv_file: Path to the input CSV (combined_crypto_scores.csv)
        json_file: Path to the JSON file (user_growth_visual_minimal.json)
        output_csv_file: Path to save the updated CSV
        json_column_name: Name of the new column to store JSON data
    """
    print(f"Loading CSV from {csv_file}...")
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} projects from CSV")
    
    print(f"Loading JSON from {json_file}...")
    # Load JSON
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} not found")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Extract protocols from JSON
    protocols = json_data.get('protocols', {})
    print(f"Loaded {len(protocols)} protocols from JSON")
    
    # Rename columns to match Google Sheet
    column_mapping = {
        'Project': 'Project',
        'Market Sector': 'Market Sector',
        'Earnings Quality Score': 'EQS',
        'User Growth Score': 'UGS',
        'Fair Value Score': 'FVS',
        'Safety Score': 'SS',
        'Growth Category': 'Growth Category',
        'Valuation Category': 'Valuation Category',
        'Safety Grade': 'SS (Rating)'
    }
    
    # Create new DataFrame with Google Sheet columns
    google_sheet_columns = [
        'Project', 'Market Sector', 'EQS', 'UGS', 'FVS', 'SS', 'how3 Score',
        'Growth Category', 'Valuation Category', 'SS (Rating)',
        'UGS dune', 'EQS dune', 'Static', 'Live', 'Dune Links', 'Symbol',
        json_column_name
    ]
    
    # Initialize new DataFrame
    new_df = pd.DataFrame(columns=google_sheet_columns)
    
    # Copy and rename existing columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            new_df[new_col] = df[old_col]
    
    # Drop Market Cap to Revenue Ratio (not in Google Sheet)
    # (No action needed, as it's not included in column_mapping)
    
    # Add new columns with default values
    new_df['how3 Score'] = ''  # Empty, as calculation is unknown
    new_df['UGS dune'] = False
    new_df['EQS dune'] = False
    new_df['Static'] = True
    new_df['Live'] = False
    new_df['Dune Links'] = ''
    new_df['Symbol'] = ''  # Could add a mapping if provided
    new_df[json_column_name] = ''
    
    # Optional: Symbol mapping (example, extend as needed)
    symbol_mapping = {
        'Convex Finance': 'cvx',
        'Aave': 'aave',
        'Algorand': 'algo',
        # Add more mappings if you have them
    }
    new_df['Symbol'] = new_df['Project'].map(symbol_mapping).fillna('')
    
    # Populate JSON column
    matched_projects = 0
    for index, row in new_df.iterrows():
        project_name = row['Project']
        if project_name in protocols:
            # Convert the protocol's JSON object to a string
            json_str = json.dumps(protocols[project_name], ensure_ascii=False)
            new_df.at[index, json_column_name] = json_str
            matched_projects += 1
        else:
            # Leave empty string for projects not in JSON
            pass
    
    print(f"Matched {matched_projects} projects with JSON data")
    
    # Save updated CSV
    print(f"Saving updated CSV to {output_csv_file}...")
    new_df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"Updated CSV saved successfully")
    
    # Print sample of updated data
    print("\nSample of updated CSV (first 5 rows):")
    print(new_df[['Project', 'Market Sector', 'UGS', json_column_name]].head(5))

if __name__ == "__main__":
    # File paths
    csv_file = "combined_crypto_scores.csv"
    json_file = "user_growth_visual_minimal.json"
    output_csv_file = "combined_crypto_scores_with_json.csv"
    
    # Check if input files exist
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found")
        exit(1)
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file {json_file} not found")
        exit(1)
    
    # Add JSON column and transform to Google Sheet format
    add_json_column_to_csv(csv_file, json_file, output_csv_file)