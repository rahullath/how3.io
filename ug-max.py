import pandas as pd
import json
import os
from collections import defaultdict

def generate_ug_datasets(csv_file, complete_output_file, minimal_output_file, focus_protocols):
    """
    Generate both complete and minimal User Growth datasets from the CSV file.
    
    Args:
        csv_file: Path to the combined_crypto_scores.csv file
        complete_output_file: Where to save the complete JSON data
        minimal_output_file: Where to save the minimal JSON data for visualization
        focus_protocols: List of protocols to focus on for the minimal dataset
    """
    print(f"Loading user growth data from {csv_file}...")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} projects from CSV")
    
    # Create complete dataset
    complete_data = {}
    
    for _, row in df.iterrows():
        project_name = row['Project']
        sector = row['Market Sector']
        
        # Extract user growth score and category
        ug_score = row.get('User Growth Score')
        growth_category = row.get('Growth Category')
        
        # Convert to numeric and handle NaN or empty strings
        try:
            ug_score = float(ug_score) if pd.notna(ug_score) and ug_score != '' else None
        except (ValueError, TypeError):
            ug_score = None
        
        # Store in the complete dataset
        complete_data[project_name] = {
            "name": project_name,
            "sector": sector if pd.notna(sector) and sector != '' else "Unknown",
            "scores": {
                "user_growth": ug_score,
                "growth_category": growth_category if pd.notna(growth_category) and growth_category != '' else None
            }
        }
    
    # Group projects by sector for sector averages
    projects_by_sector = defaultdict(list)
    for name, data in complete_data.items():
        sector = data["sector"]
        projects_by_sector[sector].append(data)
    
    # Calculate sector averages
    sector_averages = {}
    for sector, projects in projects_by_sector.items():
        valid_projects = [p for p in projects if p["scores"]["user_growth"] is not None]
        
        if not valid_projects:
            continue
            
        total_ug = sum(p["scores"]["user_growth"] for p in valid_projects)
        count = len(valid_projects)
        
        sector_averages[sector] = {
            "user_growth": round(total_ug / count, 2) if count > 0 else None,
            "count": count
        }
    
    # Save complete dataset
    with open(complete_output_file, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"Complete dataset saved to {complete_output_file}")
    print(f"Found {len(sector_averages)} sectors")
    
    # Create minimal dataset for visualization
    minimal_data = {"protocols": {}}
    
    # Keep track of which focus protocols are found
    found_protocols = []
    
    # Check for duplicates in focus_protocols
    seen = set()
    duplicates = [p for p in focus_protocols if p in seen or seen.add(p)]
    if duplicates:
        print(f"Warning: Duplicates found in focus_protocols: {duplicates}")
    
    for name in focus_protocols:
        if name not in complete_data:
            print(f"Warning: Focus protocol '{name}' not found in dataset")
            continue
            
        found_protocols.append(name)
        protocol = complete_data[name]
        sector = protocol["sector"]
        
        # Get peers (others in same sector)
        peers = []
        
        # First add focus protocols that are peers
        for peer_name in focus_protocols:
            if peer_name != name and peer_name in complete_data and complete_data[peer_name]["sector"] == sector:
                peers.append({
                    "name": peer_name,
                    "scores": {
                        "user_growth": complete_data[peer_name]["scores"]["user_growth"]
                    }
                })
        
        # Then add other top projects from the same sector
        other_projects = [
            p for p in projects_by_sector[sector] 
            if p["name"] not in focus_protocols and p["name"] != name
            and p["scores"]["user_growth"] is not None
        ]
        
        # Sort by User Growth score
        other_projects.sort(key=lambda x: x["scores"]["user_growth"] or 0, reverse=True)
        
        # Add top projects to reach 4 total peers
        needed = 4 - len(peers)
        for p in other_projects[:needed]:
            peers.append({
                "name": p["name"],
                "scores": {
                    "user_growth": p["scores"]["user_growth"]
                }
            })
        
        # Add to minimal dataset
        minimal_data["protocols"][name] = {
            "name": name,
            "sector": sector,
            "scores": protocol["scores"],
            "peers": peers[:4],
            "sector_averages": sector_averages.get(sector, {
                "user_growth": None,
                "count": 0
            })
        }
    
    # Add sector metadata
    minimal_data["sectors"] = {
        sector: {
            "name": sector,
            "averages": averages,
            "protocol_count": averages["count"]
        }
        for sector, averages in sector_averages.items()
    }
    
    # Save minimal dataset
    with open(minimal_output_file, 'w') as f:
        json.dump(minimal_data, f, indent=2)
    
    print(f"Minimal dataset saved to {minimal_output_file}")
    print(f"Included {len(found_protocols)} of {len(focus_protocols)} focus protocols")
    
    # Print summary for verification
    print("\nSector averages:")
    for sector, averages in sorted(sector_averages.items(), key=lambda x: x[1]["user_growth"] or 0, reverse=True):
        if averages["user_growth"] is not None:
            print(f"{sector}: User Growth={averages['user_growth']:.2f} (from {averages['count']} projects)")
    
    return found_protocols

if __name__ == "__main__":
    # File paths
    csv_file = "combined_crypto_scores.csv"
    complete_output_file = "complete_ugs_data.json"
    minimal_output_file = "user_growth_visual_minimal.json"
    
    # Load CSV to dynamically generate focus_protocols
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found")
        exit(1)
    
    df = pd.read_csv(csv_file)
    # Select projects with no empty columns (all 10 columns must have non-empty, non-NaN values)
    required_columns = [
        'Project', 'Market Sector', 'Safety Score', 'Safety Grade',
        'Earnings Quality Score', 'User Growth Score', 'Growth Category',
        'Fair Value Score', 'Valuation Category', 'Market Cap to Revenue Ratio'
    ]
    # Check for non-empty, non-NaN values, treating "0.0x" as valid
    focus_protocols = df[
        df[required_columns].notna().all(axis=1) & 
        (df[required_columns].ne('').all(axis=1))
    ]['Project'].tolist()
    focus_protocols = sorted(list(set(focus_protocols)))  # Remove duplicates and sort
    print(f"Generated {len(focus_protocols)} focus protocols with no empty columns:")
    for protocol in focus_protocols:
        print(f"- {protocol}")
    
    # Generate both datasets
    found_protocols = generate_ug_datasets(
        csv_file, 
        complete_output_file, 
        minimal_output_file,
        focus_protocols
    )
    
    # Print which focus protocols were found and used
    print("\nFocus protocols included in the visualization:")
    for protocol in sorted(found_protocols):
        print(f"- {protocol}")
    
    missing = set(focus_protocols) - set(found_protocols)
    if missing:
        print("\nWarning: These focus protocols were not found in the dataset:")
        for protocol in sorted(missing):
            print(f"- {protocol}")