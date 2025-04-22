import pandas as pd
import json
import os
from collections import defaultdict

def generate_relative_value_jsons(eqs_csv_path, ugs_csv_path, output_dir):
    """
    Generate separate JSON files for each category for the relative value chart.
    
    Args:
        eqs_csv_path: Path to the earnings quality score CSV file
        ugs_csv_path: Path to the user growth score CSV file
        output_dir: Directory where to save the output JSON files
    """
    print(f"Loading data from {eqs_csv_path} and {ugs_csv_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV files
    eqs_df = pd.read_csv(eqs_csv_path)
    ugs_df = pd.read_csv(ugs_csv_path)
    
    print(f"Loaded {len(eqs_df)} projects from EQS CSV")
    print(f"Loaded {len(ugs_df)} projects from UGS CSV")
    
    # Define the refined sector mapping
    sector_mapping = {
        'blockchains (l1)': 'Layer 1',
        'blockchains (l2)': 'Layer 2',
        'blockchains (l3)': 'Layer 2',  # Group L3 with L2
        'exchanges (dex)': 'DeFi',
        'lending': 'DeFi',
        'liquid staking': 'DeFi',
        'derivative exchanges': 'DeFi',
        'asset management': 'DeFi',
        'exchanges (cex)': 'CeFi',
        'gaming': 'GameFi',
        'nft marketplaces': 'GameFi',
        'infrastructure': 'Infrastructure',
        'bridges': 'Infrastructure',
        'oracles': 'Infrastructure',
        'real world assets': 'RWA',
        'rwa': 'RWA',
        'ai': 'AI',
        'artificial intelligence': 'AI',
    }
    
    # Project symbol mapping - extend as needed
    project_symbols = {
        'Bitcoin': 'BTC',
        'Ethereum': 'ETH',
        'Tether': 'USDT',
        'BNB Chain': 'BNB',
        'Solana': 'SOL',
        'XRP': 'XRP',
        'Cardano': 'ADA',
        'Dogecoin': 'DOGE',
        'Avalanche': 'AVAX',
        'Polkadot': 'DOT',
        'Uniswap': 'UNI',
        'Lido Finance': 'LDO',
        'Chainlink': 'LINK',
        'Tron': 'TRX',
        'Circle': 'USDC',
        'Aave': 'AAVE',
        'GMX': 'GMX',
        'Uniswap Labs': 'UNI',
        'OP Mainnet': 'OP',
        'Pendle': 'PENDLE',
        'Curve': 'CRV',
        'Maple Finance': 'MPL',
        'Compound': 'COMP',
    }
    
    # Process EQS data
    eqs_projects = {}
    for _, row in eqs_df.iterrows():
        project_name = row['Project']
        sector = row['Market Sector'] if pd.notna(row['Market Sector']) else 'Unknown'
        
        # Skip sectors not in our mapping
        sector_lower = sector.lower()
        if sector_lower not in sector_mapping:
            continue
            
        simplified_sector = sector_mapping[sector_lower]
        
        # Get revenue quality score
        try:
            eqs_score = float(row['Revenue Quality Score']) if pd.notna(row['Revenue Quality Score']) else None
        except (ValueError, TypeError):
            eqs_score = None

        if eqs_score is None:
            continue  # Skip if EQS score is null
        
        # Store in the data dictionary
        eqs_projects[project_name] = {
            'project': project_name,
            'symbol': project_symbols.get(project_name, project_name[:3].upper()),  # Use first 3 chars if no mapping
            'position': {
                'x': round(eqs_score, 2),
                'y': 0  # Will be updated if UGS data exists
            },
            'meta': {
                'sector': simplified_sector,
                'original_sector': sector
            },
            'highlight': False
        }
    
    # Process UGS data and update the projects
    for _, row in ugs_df.iterrows():
        project_name = row['Project']
        
        # If project exists in EQS data, update it
        if project_name in eqs_projects:
            try:
                ugs_score = float(row['User Growth Score']) if pd.notna(row['User Growth Score']) else None
                if ugs_score is not None:
                    eqs_projects[project_name]['position']['y'] = round(ugs_score, 2)
            except (ValueError, TypeError):
                pass  # Skip if can't convert to float
        else:
            # For projects only in UGS data, check if the sector is in our mapping
            sector = row['Market Sector'] if pd.notna(row['Market Sector']) else 'Unknown'
            sector_lower = sector.lower()
            
            if sector_lower not in sector_mapping:
                continue
                
            simplified_sector = sector_mapping[sector_lower]
            
            try:
                ugs_score = float(row['User Growth Score']) if pd.notna(row['User Growth Score']) else None
                if ugs_score is None:
                    continue  # Skip if UGS score is null
                
                eqs_projects[project_name] = {
                    'project': project_name,
                    'symbol': project_symbols.get(project_name, project_name[:3].upper()),
                    'position': {
                        'x': 0.0,  # No EQS data
                        'y': round(ugs_score, 2)
                    },
                    'meta': {
                        'sector': simplified_sector,
                        'original_sector': sector
                    },
                    'highlight': False
                }
            except (ValueError, TypeError):
                pass  # Skip if can't convert to float
    
    # Filter out projects with no meaningful data (both scores are 0 or null)
    filtered_projects = {
        name: data for name, data in eqs_projects.items()
        if data['position']['x'] > 0 or data['position']['y'] > 0
    }
    
    print(f"Found {len(filtered_projects)} projects with at least one valid score")
    
    # Group projects by sector
    sectors = defaultdict(dict)  # Use a dictionary for each sector
    for project_name, project_data in filtered_projects.items():
        sector = project_data['meta']['sector']
        sectors[sector][project_name] = project_data
    
    # Create separate JSON file for each sector
    for sector, projects in sectors.items():
        if not projects:  # Skip empty sectors
            continue

        # Limit to 20 projects per sector
        limited_projects = dict(list(projects.items())[:20])
            
        # Save to JSON file
        filename = f"{sector.lower().replace(' ', '_')}_projects.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(limited_projects, f, indent=2)
        
        print(f"Created {filename} with {len(limited_projects)} projects")
    
    # Create an "all" JSON with all projects
    all_filepath = os.path.join(output_dir, "all_projects.json")
    with open(all_filepath, 'w') as f:
        json.dump(filtered_projects, f, indent=2)
    
    print(f"Created all_projects.json with {len(filtered_projects)} projects")
    
    # Print sector summary
    print("\nSector summary:")
    for sector, projects in sectors.items():
        print(f"{sector}: {len(projects)} projects")
    
    return sectors

if __name__ == "__main__":
    # File paths - adjust these as needed
    eqs_csv_path = "earnings_quality_results.csv"
    ugs_csv_path = "user_growth_results.csv"
    output_dir = "crypto_sector_jsons"
    
    # Generate the JSON data
    sectors = generate_relative_value_jsons(eqs_csv_path, ugs_csv_path, output_dir)
    
    # Print some sample project data for verification
    print("\nSample projects by sector:")
    for sector, projects in sectors.items():
        print(f"\n{sector}:")
        sample_count = min(3, len(projects))  # Show up to 3 projects per sector
        for i, data in enumerate(list(projects.values())[:sample_count]):  # Convert to list for slicing
            print(f"  {i+1}. {data['project']} ({data['symbol']}): EQS={data['position']['x']}, UGS={data['position']['y']}")