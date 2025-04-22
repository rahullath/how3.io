#!/usr/bin/env python3
"""
Crypto Scores Combiner

This script combines data from multiple cryptocurrency analysis files:
- Safety scores from a text file
- Earnings Quality Scores (EQS) from CSV
- User Growth Scores (UGS) from CSV
- Fair Value Scores (FVS) from CSV

Usage:
    python combine_scores.py

Make sure all the required files are in the same directory as this script:
- safety score - top 330 by market cap.txt
- earnings_quality_results.csv
- user_growth_results.csv
- fair_value_results.csv
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Optional, Any

def normalize_project_name(name: str) -> str:
    """Normalize project name for better matching."""
    if not name or not isinstance(name, str):
        return ''
    
    normalized = name.lower()
    normalized = re.sub(r'\s*logo\s*', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[\(\)]', '', normalized)
    normalized = re.sub(r'chain$', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'network$', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'finance$', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'protocol$', '', normalized, flags=re.IGNORECASE)
    return normalized.strip()

def clean_value(value: Any) -> Any:
    """Clean a value from a dataframe."""
    if pd.isna(value):
        return ''
    
    if isinstance(value, str):
        # Remove line breaks
        return value.replace('\r', '').replace('\n', '')
    
    return value

def find_best_match(target_name: str, data_df: pd.DataFrame, name_col: str) -> Optional[pd.Series]:
    """Find the best match for a project name in a dataframe."""
    if not target_name or data_df.empty:
        return None
    
    # Normalize the target name
    normalized_target = normalize_project_name(target_name)
    if not normalized_target:
        return None
    
    # Normalize all names in the dataframe
    data_df['normalized_name'] = data_df[name_col].apply(normalize_project_name)
    
    # Try exact match first
    exact_matches = data_df[data_df['normalized_name'] == normalized_target]
    if not exact_matches.empty:
        return exact_matches.iloc[0]
    
    # Try partial matches
    partial_matches = data_df[
        data_df['normalized_name'].str.contains(normalized_target, regex=False) | 
        data_df['normalized_name'].apply(lambda x: normalized_target in x if x else False)
    ]
    
    if len(partial_matches) == 1:
        return partial_matches.iloc[0]
    
    if len(partial_matches) > 1:
        # Find closest match by length difference
        partial_matches['length_diff'] = partial_matches['normalized_name'].apply(
            lambda x: abs(len(x) - len(normalized_target))
        )
        return partial_matches.sort_values('length_diff').iloc[0]
    
    return None

def parse_safety_scores(file_path: str) -> pd.DataFrame:
    """Parse the safety score text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        projects = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(r'^\d+$', line) and i + 4 < len(lines):
                rank = int(line)
                logo_line = lines[i + 1].strip()
                name_line = lines[i + 2].strip()
                score_line = lines[i + 3].strip()
                grade_line = lines[i + 4].strip()
                
                if ('logo' in logo_line and 
                    name_line and 
                    re.match(r'^\d+\.\d+$', score_line) and 
                    re.match(r'^[A-D]{1,3}$', grade_line)):
                    
                    projects.append({
                        'rank': rank,
                        'name': name_line,
                        'safety_score': float(score_line),
                        'grade': grade_line
                    })
                    
                    i += 9  # Skip to the next project entry
                    continue
            
            i += 1
        
        return pd.DataFrame(projects)
    except Exception as e:
        print(f"Error parsing safety scores: {e}")
        return pd.DataFrame()

def combine_crypto_scores():
    """Combine scores from different CSV files."""
    # Check if files exist
    required_files = [
        'safety score - top 330 by market cap.txt',
        'earnings_quality_results.csv',
        'user_growth_results.csv',
        'fair_value_results.csv'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return None
    
    # Parse safety scores
    safety_df = parse_safety_scores('safety score - top 330 by market cap.txt')
    print(f"Parsed {len(safety_df)} safety scores")
    
    # Load CSV files
    try:
        eqs_df = pd.read_csv('earnings_quality_results.csv')
        print(f"Loaded {len(eqs_df)} earnings quality results")
    except Exception as e:
        print(f"Error loading earnings quality results: {e}")
        eqs_df = pd.DataFrame()
        
    try:
        ugs_df = pd.read_csv('user_growth_results.csv')
        print(f"Loaded {len(ugs_df)} user growth results")
    except Exception as e:
        print(f"Error loading user growth results: {e}")
        ugs_df = pd.DataFrame()
        
    try:
        fvs_df = pd.read_csv('fair_value_results.csv')
        print(f"Loaded {len(fvs_df)} fair value results")
    except Exception as e:
        print(f"Error loading fair value results: {e}")
        fvs_df = pd.DataFrame()
    
    # Create a set of all unique project names (normalized)
    all_projects = set()
    
    # Add projects from safety scores
    if not safety_df.empty and 'name' in safety_df.columns:
        safety_df['normalized_name'] = safety_df['name'].apply(normalize_project_name)
        all_projects.update(safety_df['normalized_name'].dropna().tolist())
    
    # Add projects from earnings quality
    if not eqs_df.empty and 'Project' in eqs_df.columns:
        eqs_df['normalized_name'] = eqs_df['Project'].apply(normalize_project_name)
        all_projects.update(eqs_df['normalized_name'].dropna().tolist())
    
    # Add projects from user growth
    if not ugs_df.empty and 'Project' in ugs_df.columns:
        ugs_df['normalized_name'] = ugs_df['Project'].apply(normalize_project_name)
        all_projects.update(ugs_df['normalized_name'].dropna().tolist())
    
    # Add projects from fair value
    if not fvs_df.empty and 'Project' in fvs_df.columns:
        fvs_df['normalized_name'] = fvs_df['Project'].apply(normalize_project_name)
        all_projects.update(fvs_df['normalized_name'].dropna().tolist())
    
    print(f"Found {len(all_projects)} unique normalized project names")
    
    # Create combined data
    combined_data = []
    
    for norm_name in all_projects:
        if not norm_name:
            continue
        
        # Find matches in each dataset
        safety_match = None
        if not safety_df.empty:
            safety_match = find_best_match(norm_name, safety_df, 'name')
        
        eqs_match = None
        if not eqs_df.empty:
            eqs_match = find_best_match(norm_name, eqs_df, 'Project')
        
        ugs_match = None
        if not ugs_df.empty:
            ugs_match = find_best_match(norm_name, ugs_df, 'Project')
        
        fvs_match = None
        if not fvs_df.empty:
            fvs_match = find_best_match(norm_name, fvs_df, 'Project')
        
        # Get the best project name to use
        project_name = ""
        if safety_match is not None and 'name' in safety_match:
            project_name = safety_match['name']
        elif eqs_match is not None and 'Project' in eqs_match:
            project_name = eqs_match['Project']
        elif ugs_match is not None and 'Project' in ugs_match:
            project_name = ugs_match['Project']
        elif fvs_match is not None and 'Project' in fvs_match:
            project_name = fvs_match['Project']
        
        if not project_name:
            continue
        
        # Get market sector
        market_sector = ""
        if eqs_match is not None and 'Market Sector' in eqs_match:
            market_sector = eqs_match['Market Sector']
        elif ugs_match is not None and 'Market Sector' in ugs_match:
            market_sector = ugs_match['Market Sector']
        elif fvs_match is not None and 'Market Sector' in fvs_match:
            market_sector = fvs_match['Market Sector']
        
        # Create combined record
        record = {
            'Project': clean_value(project_name),
            'Market Sector': clean_value(market_sector),
            'Safety Score': clean_value(safety_match['safety_score'] if safety_match is not None else ''),
            'Safety Grade': clean_value(safety_match['grade'] if safety_match is not None else ''),
            'Earnings Quality Score': clean_value(eqs_match['Revenue Quality Score'] if eqs_match is not None and 'Revenue Quality Score' in eqs_match else ''),
            'User Growth Score': clean_value(ugs_match['User Growth Score'] if ugs_match is not None and 'User Growth Score' in ugs_match else ''),
            'Growth Category': clean_value(ugs_match['Growth Category'] if ugs_match is not None and 'Growth Category' in ugs_match else ''),
            'Fair Value Score': clean_value(fvs_match['Fair Value Score'] if fvs_match is not None and 'Fair Value Score' in fvs_match else ''),
            'Valuation Category': clean_value(fvs_match['Valuation Category'] if fvs_match is not None and 'Valuation Category' in fvs_match else ''),
            'Market Cap to Revenue Ratio': clean_value(fvs_match['Market Cap to Revenue Ratio'] if fvs_match is not None and 'Market Cap to Revenue Ratio' in fvs_match else '')
        }
        
        combined_data.append(record)
    
    # Convert to DataFrame and sort by project name
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values('Project')
    
    print(f"Created {len(combined_df)} combined records")
    
    # Save to CSV
    output_file = 'combined_crypto_scores.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting data combination process...")
    combined_df = combine_crypto_scores()
    
    if combined_df is not None:
        # Print first few rows
        print("\nSample of combined data:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(combined_df.head(5))
        
        # Print some statistics
        print("\nData statistics:")
        safety_count = combined_df['Safety Score'].notna().sum()
        eqs_count = combined_df['Earnings Quality Score'].notna().sum()
        ugs_count = combined_df['User Growth Score'].notna().sum()
        fvs_count = combined_df['Fair Value Score'].notna().sum()
        
        print(f"Total projects: {len(combined_df)}")
        print(f"Projects with Safety Score: {safety_count} ({safety_count/len(combined_df)*100:.1f}%)")
        print(f"Projects with Earnings Quality Score: {eqs_count} ({eqs_count/len(combined_df)*100:.1f}%)")
        print(f"Projects with User Growth Score: {ugs_count} ({ugs_count/len(combined_df)*100:.1f}%)")
        print(f"Projects with Fair Value Score: {fvs_count} ({fvs_count/len(combined_df)*100:.1f}%)")
        
        print(f"\nComplete! All data has been combined into '{os.path.abspath('combined_crypto_scores.csv')}'")
    else:
        print("Process failed. Please check the error messages above.")