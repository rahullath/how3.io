import pandas as pd
import json
import os

def add_json_column_to_csv(csv_file, eqs_json_file, output_csv_file, eqs_column_name='EQSJSON'):
    """
    Add EQSJSON and optionally UserGrowthJSON columns to the CSV, matching the Google Sheet format.
    
    Args:
        csv_file: Path to the input CSV (combined_crypto_scores.csv)
        eqs_json_file: Path to the EQS JSON file (eqs_visualization_minimal.json)
        ugs_json_file: Path to the UGS JSON file (user_growth_visualization_minimal.json, optional)
        output_csv_file: Path to save the updated CSV
        eqs_column_name: Name of the EQS JSON column
        ugs_column_name: Name of the UGS JSON column (optional)
    """
    print(f"Loading CSV from {csv_file}...")
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} projects from CSV")
    
    # Override sector for Sky (formerly MakerDAO)
    df.loc[df['Project'] == 'Sky (formerly MakerDAO)', 'Market Sector'] = 'Lending'
    
    # Load EQS JSON
    print(f"Loading EQS JSON from {eqs_json_file}...")
    if not os.path.exists(eqs_json_file):
        print(f"Error: EQS JSON file {eqs_json_file} not found")
        return
    with open(eqs_json_file, 'r', encoding='utf-8') as f:
        eqs_json_data = json.load(f)
    eqs_protocols = eqs_json_data.get('protocols', {})
    print(f"Loaded {len(eqs_protocols)} protocols from EQS JSON")

    
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
        'UGS dune', 'EQS dune', 'Static', 'Live', 'Dune Links', 'Symbol'
    ]
    if eqs_column_name:
        google_sheet_columns.append(eqs_column_name)
    
    # Initialize new DataFrame
    new_df = pd.DataFrame(columns=google_sheet_columns)
    
    # Copy and rename existing columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            new_df[new_col] = df[old_col]
    
    # Add new columns with default values

    
    # Optional: Symbol mapping
    symbol_mapping = {
        'Convex Finance': 'cvx',
        'Aave': 'aave',
        'Algorand': 'algo',
        'Aptos': 'apt',
        'Avalanche': 'avax',
        # Add more if available
    }
    new_df['Symbol'] = new_df['Project'].map(symbol_mapping).fillna('')
    
    # Populate EQSJSON column
    if eqs_column_name:
        new_df[eqs_column_name] = ''
        matched_eqs = 0
        for index, row in new_df.iterrows():
            project_name = row['Project']
            if project_name in eqs_protocols:
                json_str = json.dumps(eqs_protocols[project_name], ensure_ascii=False)
                new_df.at[index, eqs_column_name] = json_str
                matched_eqs += 1
    
    
    print(f"Matched {matched_eqs} projects with EQS JSON data")
   
    # Save updated CSV
    print(f"Saving updated CSV to {output_csv_file}...")
    new_df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"Updated CSV saved successfully")
    
    # Print sample of updated data
    print("\nSample of updated CSV (first 5 rows):")
    sample_columns = ['Project', 'Market Sector', 'EQS']
    if eqs_column_name:
        sample_columns.append(eqs_column_name)
    print(new_df[sample_columns].head(5))

if __name__ == "__main__":
    # File paths
    csv_file = "combined_crypto_scores.csv"
    eqs_json_file = "eqs_visualization_minimal.json"
    output_csv_file = "eqsjson.csv"
    
    # Check if input files exist
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found")
        exit(1)
    
    if not os.path.exists(eqs_json_file):
        print(f"Error: EQS JSON file {eqs_json_file} not found")
        exit(1)
    
    
    # Add JSON columns
    add_json_column_to_csv(csv_file, eqs_json_file,output_csv_file)