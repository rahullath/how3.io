import pandas as pd
import json
import ast
import os
from collections import defaultdict

def generate_eqs_datasets(csv_file, complete_output_file, minimal_output_file, focus_protocols):
    """
    Generate both complete and minimal EQS datasets from the CSV file.
    
    Args:
        csv_file: Path to the earnings_quality_result.csv file
        complete_output_file: Where to save the complete JSON data
        minimal_output_file: Where to save the minimal JSON data for visualization
        focus_protocols: List of protocols to focus on for the minimal dataset
    """
    print(f"Loading data from {csv_file}...")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} projects from CSV")
    
    # Parse the Quality Explanation column which contains JSON-like strings
    def parse_explanation(explanation_str):
        try:
            # Convert the string to a proper dictionary
            # First, replace single quotes with double quotes for JSON compatibility
            json_str = explanation_str.replace("'", "\"")
            return json.loads(json_str)
        except:
            try:
                # If JSON parsing fails, try using ast.literal_eval
                return ast.literal_eval(explanation_str)
            except:
                print(f"Error parsing: {explanation_str}")
                return {
                    "stability_score": None,
                    "magnitude_score": None,
                    "revenue_quality_score": None,
                    "method": "Unknown"
                }
    
    # Create complete dataset
    complete_data = {}
    
    for _, row in df.iterrows():
        project_name = row['Project']
        sector = row['Market Sector']
        
        # Parse the explanation field
        explanation = parse_explanation(row['Quality Explanation'])
        
        # Extract scores
        stability_score = explanation.get('stability_score')
        magnitude_score = explanation.get('magnitude_score')
        eqs_score = explanation.get('revenue_quality_score')
        
        # Store in the complete dataset
        complete_data[project_name] = {
            "name": project_name,
            "sector": sector,
            "scores": {
                "earnings_quality": eqs_score,
                "stability": stability_score,
                "magnitude": magnitude_score
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
        valid_projects = [p for p in projects if p["scores"]["earnings_quality"] is not None]
        
        if not valid_projects:
            continue
            
        total_eqs = sum(p["scores"]["earnings_quality"] for p in valid_projects if p["scores"]["earnings_quality"] is not None)
        total_stability = sum(p["scores"]["stability"] for p in valid_projects if p["scores"]["stability"] is not None)
        total_magnitude = sum(p["scores"]["magnitude"] for p in valid_projects if p["scores"]["magnitude"] is not None)
        count = len(valid_projects)
        
        sector_averages[sector] = {
            "earnings_quality": round(total_eqs / count, 2) if count > 0 else None,
            "stability": round(total_stability / count, 2) if count > 0 else None,
            "magnitude": round(total_magnitude / count, 2) if count > 0 else None,
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
                        "earnings_quality": complete_data[peer_name]["scores"]["earnings_quality"]
                    }
                })
        
        # Then add other top projects from the same sector
        other_projects = [
            p for p in projects_by_sector[sector] 
            if p["name"] not in focus_protocols and p["name"] != name
            and p["scores"]["earnings_quality"] is not None
        ]
        
        # Sort by EQS score
        other_projects.sort(key=lambda x: x["scores"]["earnings_quality"] or 0, reverse=True)
        
        # Add top projects to reach 4 total peers
        needed = 4 - len(peers)
        for p in other_projects[:needed]:
            peers.append({
                "name": p["name"],
                "scores": {
                    "earnings_quality": p["scores"]["earnings_quality"]
                }
            })
        
        # Add to minimal dataset
        minimal_data["protocols"][name] = {
            "name": name,
            "sector": sector,
            "scores": protocol["scores"],
            "peers": peers[:4],
            "sector_averages": sector_averages.get(sector, {
                "earnings_quality": None,
                "stability": None,
                "magnitude": None,
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
    for sector, averages in sorted(sector_averages.items(), key=lambda x: x[1]["earnings_quality"] or 0, reverse=True):
        if averages["earnings_quality"] is not None:
            print(f"{sector}: EQS={averages['earnings_quality']:.2f} (from {averages['count']} projects)")
    
    return found_protocols

if __name__ == "__main__":
    # File paths
    csv_file = "earnings_quality_results.csv"
    complete_output_file = "complete_eqs_data.json"
    minimal_output_file = "eqs_visualization_minimal.json"
    
    # List of 14 focus protocols for the visualization
    focus_protocols = [
        "Aave", "GMX", "Lido Finance", "Bitcoin", "Uniswap Labs", 
        "BNB Chain", "OP Mainnet", "Pendle", "Curve", 
        "Maple Finance", "Compound", "Ethereum", "Avalanche", "Chainlink"
    ]
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found")
        exit(1)
    
    # Generate both datasets
    found_protocols = generate_eqs_datasets(
        csv_file, 
        complete_output_file, 
        minimal_output_file,
        focus_protocols
    )
    
    # Print which focus protocols were found and used
    print("\nFocus protocols included in the visualization:")
    for protocol in found_protocols:
        print(f"- {protocol}")
    
    missing = set(focus_protocols) - set(found_protocols)
    if missing:
        print("\nWarning: These focus protocols were not found in the dataset:")
        for protocol in missing:
            print(f"- {protocol}")