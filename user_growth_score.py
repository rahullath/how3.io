import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import argparse

class UserGrowthAnalyzer:
    """
    A specialized class for analyzing user growth and adoption metrics
    for cryptocurrency projects.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe containing crypto project metrics."""
        self.df = df
        self._clean_data()
        self.growth_columns = self._find_growth_columns()
        self.sector_metrics = self._get_market_sector_metrics()
    
    def _clean_data(self) -> None:
        """Clean and prepare the data for analysis."""
        if self.df is None:
            return
        
        # Replace infinity values with NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Standardize key column names
        column_mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            if col_lower == 'project' or 'project' in col_lower and len(col_lower) < 15:
                column_mapping[col] = 'Project'
            elif col_lower == 'market sector' or 'market sector' in col_lower or 'sector' in col_lower:
                column_mapping[col] = 'Market sector'
            elif col_lower == 'listing date':
                column_mapping[col] = 'Listing Date'
        
        # Rename columns if needed
        if column_mapping:
            self.df = self.df.rename(columns=column_mapping)
            print(f"Renamed {len(column_mapping)} columns for standardization")
        
        # Check for tuple/multi-level columns
        if isinstance(self.df.columns, pd.MultiIndex):
            print("Detected multi-level column headers, flattening...")
            # Flatten the column names
            self.df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in self.df.columns]
            print("Column headers flattened")
        
        # Handle unnamed columns
        unnamed_cols = [col for col in self.df.columns if 'Unnamed:' in str(col)]
        if unnamed_cols:
            print(f"Found {len(unnamed_cols)} unnamed columns, checking first row for headers...")
            # Check if first row contains headers
            if len(self.df) > 0:
                first_row = self.df.iloc[0]
                header_row = True
                for col in unnamed_cols:
                    val = first_row[col]
                    if pd.isna(val) or not isinstance(val, str):
                        header_row = False
                        break
                
                if header_row:
                    print("First row appears to contain column headers, using them...")
                    # Create a mapping of unnamed columns to their header values
                    header_mapping = {col: first_row[col] for col in unnamed_cols if isinstance(first_row[col], str)}
                    # Rename columns
                    self.df = self.df.rename(columns=header_mapping)
                    # Drop the header row
                    self.df = self.df.iloc[1:].reset_index(drop=True)
        
        # Convert numeric columns where possible
        for col in self.df.columns:
            if col not in ['Project', 'Market sector', 'Listing Date']:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass
    
    def _find_growth_columns(self) -> Dict[str, List[str]]:
        """Find columns related to user growth metrics."""
        growth_columns = {
            'active_users': [],
            'transaction_volume': [],
            'bridge_volume': [],
            'transaction_count': [],
            'user_growth': []
        }
        
        # Map column name patterns to metrics
        column_patterns = {
            'active_users': [
                'Active users', 'Active addresses', 'Daily active', 'DAU', 
                'Monthly active', 'MAU', 'Weekly active', 'WAU'
            ],
            'transaction_volume': [
                'Transaction volume', 'Trading volume', 'Transfer volume', 
                'Volume', 'Notional'
            ],
            'bridge_volume': [
                'Bridge deposits', 'Bridge volume', 'Net deposits'
            ],
            'transaction_count': [
                'Transaction count', 'Transactions per', 'Trade count', 
                'Number of transactions'
            ],
            'user_growth': [
                'User growth', 'User adoption', 'Growth rate', 'User increase',
                'Stablecoin holders', 'Tokenholders'
            ]
        }
        
        # Search for columns matching patterns
        for metric, patterns in column_patterns.items():
            for col in self.df.columns:
                col_str = str(col).lower()
                if any(pattern.lower() in col_str for pattern in patterns):
                    growth_columns[metric].append(col)
        
        # Log found columns
        for metric, cols in growth_columns.items():
            print(f"Found {len(cols)} columns for {metric}: {cols[:3]}")
            
        return growth_columns
    
    def _get_market_sector_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Define which metrics to prioritize for each market sector.
        Returns a dictionary of sector -> metric -> weight mappings.
        """
        sector_metrics = {
            'Lending': {
                'active_users': 0.40,
                'transaction_volume': 0.30,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'Exchanges (DEX)': {
                'active_users': 0.30,
                'transaction_volume': 0.40,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'Derivative exchanges': {
                'active_users': 0.30,
                'transaction_volume': 0.40,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'Blockchains (L1)': {
                'active_users': 0.30,
                'transaction_volume': 0.30,
                'transaction_count': 0.25,
                'user_growth': 0.15
            },
            'Blockchains (L2)': {
                'active_users': 0.30,
                'transaction_volume': 0.25,
                'bridge_volume': 0.20,
                'transaction_count': 0.15,
                'user_growth': 0.10
            },
            'Bridges': {
                'active_users': 0.20,
                'bridge_volume': 0.50,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'NFT marketplaces': {
                'active_users': 0.40,
                'transaction_volume': 0.30,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'Liquid staking': {
                'active_users': 0.30,
                'transaction_volume': 0.20,
                'user_growth': 0.30,
                'transaction_count': 0.20
            },
            'Stablecoin issuers': {
                'active_users': 0.20,
                'transaction_volume': 0.40,
                'transaction_count': 0.20,
                'user_growth': 0.20
            },
            'Infrastructure': {
                'active_users': 0.30,
                'transaction_volume': 0.30,
                'transaction_count': 0.30,
                'user_growth': 0.10
            },
            'Gaming': {
                'active_users': 0.50,
                'transaction_volume': 0.20,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            'Social': {
                'active_users': 0.60,
                'transaction_volume': 0.10,
                'transaction_count': 0.10,
                'user_growth': 0.20
            },
            'Asset management': {
                'active_users': 0.30,
                'transaction_volume': 0.40,
                'transaction_count': 0.20,
                'user_growth': 0.10
            },
            # Default weights for any other sector
            'default': {
                'active_users': 0.35,
                'transaction_volume': 0.30,
                'transaction_count': 0.20,
                'user_growth': 0.15
            }
        }
        
        return sector_metrics
        
    def _get_best_column(self, metric: str, prefer_latest: bool = True) -> Optional[str]:
        """
        Get the most appropriate column for a specific metric.
        
        Args:
            metric: The metric category ('active_users', 'transaction_volume', etc.)
            prefer_latest: Whether to prefer columns with 'latest' or 'current' data
            
        Returns:
            The column name or None if no suitable column found
        """
        if metric not in self.growth_columns or not self.growth_columns[metric]:
            return None
            
        columns = self.growth_columns[metric]
        
        # Preferred time periods in order
        periods = ['Latest', '24h', '7d', '30d', '90d', '180d', '365d']
        
        if prefer_latest:
            # Try to find columns matching preferred periods
            for period in periods:
                for col in columns:
                    col_str = str(col).lower()
                    if period.lower() in col_str:
                        return col
        
        # If no preferred period found, return the first column
        return columns[0]
    
    def _get_numeric_value(self, value: Any) -> Optional[float]:
        """Convert a value to a numeric value safely."""
        if pd.isna(value):
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point
            try:
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            except:
                return None
                
        return None
        
    def _get_value(self, row_idx: int, col_name: str, default_val: Any = None) -> Any:
        """Safely get a value from the dataframe."""
        if col_name not in self.df.columns:
            return default_val
            
        value = self.df.iloc[row_idx][col_name]
        
        if pd.isna(value):
            return default_val
            
        return value
    
    def calculate_metric_score(self, row_idx: int, metric: str, sector: str) -> Tuple[Optional[float], Dict]:
        """
        Calculate a score for a specific growth metric.
        
        Args:
            row_idx: Index of the project in the dataframe
            metric: Metric to calculate ('active_users', 'transaction_volume', etc.)
            sector: Market sector of the project
            
        Returns:
            Tuple of (score, explanation_dict)
        """
        explanation = {
            'metric': metric,
            'column_used': None,
            'value': None,
            'percentile': None,
            'score': None,
            'sector_comparison': None,
            'num_comparisons': 0
        }
        
        # Get the best column for this metric
        col_name = self._get_best_column(metric)
        if not col_name:
            return None, explanation
            
        explanation['column_used'] = col_name
        
        # Get the value for this project
        value = self._get_value(row_idx, col_name)
        numeric_value = self._get_numeric_value(value)
        
        explanation['value'] = value
        explanation['numeric_value'] = numeric_value
        
        if numeric_value is None:
            return None, explanation
            
        # Get the sector filter
        sector_mask = self.df['Market sector'] == sector
        sector_projects = self.df[sector_mask]
        
        if sector_projects.empty:
            return None, explanation
            
        # Get values from all projects in the same sector
        all_values = []
        for idx, proj_row in sector_projects.iterrows():
            if idx != row_idx:  # Skip the current project
                proj_value = proj_row.get(col_name)
                proj_numeric = self._get_numeric_value(proj_value)
                if proj_numeric is not None:
                    all_values.append(proj_numeric)
        
        explanation['num_comparisons'] = len(all_values)
        
        if not all_values:
            return None, explanation
            
        # Calculate percentile (higher is better)
        percentile = sum(1 for v in all_values if v <= numeric_value) / len(all_values) * 100
        explanation['percentile'] = percentile
        
        # Convert percentile to score (0-100)
        if percentile >= 90:
            score = 90 + (percentile - 90) * (10/10)  # 90-100
        elif percentile >= 70:
            score = 70 + (percentile - 70) * (20/20)  # 70-89
        elif percentile >= 30:
            score = 40 + (percentile - 30) * (30/40)  # 40-69
        elif percentile >= 10:
            score = 20 + (percentile - 10) * (20/20)  # 20-39
        else:
            score = percentile * (20/10)  # 0-19
            
        explanation['score'] = score
        
        # Add sector comparison
        sector_avg = np.mean(all_values)
        sector_median = np.median(all_values)
        
        explanation['sector_comparison'] = {
            'average': sector_avg,
            'median': sector_median,
            'compared_to_avg': f"{(numeric_value / sector_avg - 1) * 100:.1f}%" if sector_avg > 0 else "N/A",
            'compared_to_median': f"{(numeric_value / sector_median - 1) * 100:.1f}%" if sector_median > 0 else "N/A"
        }
        
        return score, explanation
    
    def calculate_user_growth_score(self, row_idx: int) -> Dict:
        """
        Calculate the overall user growth score based on multiple metrics.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Dictionary with scores and explanations
        """
        # Get project info
        project = self.df.iloc[row_idx].get('Project', 'Unknown')
        sector = self.df.iloc[row_idx].get('Market sector', 'Unknown')
        
        # Get weights for this sector
        weights = self.sector_metrics.get(sector, self.sector_metrics['default'])
        
        # Initialize results
        results = {
            'project': project,
            'sector': sector,
            'metrics': {},
            'overall_score': None,
            'weights_used': weights,
            'explanation': "User growth score based on weighted average of key adoption metrics"
        }
        
        # Calculate score for each metric
        weighted_scores = []
        
        for metric, weight in weights.items():
            score, explanation = self.calculate_metric_score(row_idx, metric, sector)
            
            if score is not None:
                weighted_scores.append((score, weight))
                results['metrics'][metric] = {
                    'score': score,
                    'weight': weight,
                    'explanation': explanation
                }
        
        # Calculate overall score
        if weighted_scores:
            overall_score = sum(score * weight for score, weight in weighted_scores) / sum(weight for _, weight in weighted_scores)
            results['overall_score'] = overall_score
            
            # Determine growth category
            if overall_score >= 80:
                category = "Exceptional Growth"
            elif overall_score >= 65:
                category = "Strong Growth"
            elif overall_score >= 45:
                category = "Steady Growth"
            elif overall_score >= 25:
                category = "Slow Growth"
            else:
                category = "Stagnant/Declining"
                
            results['growth_category'] = category
        
        return results
    
    def analyze_all_projects(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze user growth for all projects in the dataset.
        
        Args:
            output_file: Optional path to save results CSV
            
        Returns:
            DataFrame with user growth scores
        """
        results = []
        
        # Process each project
        for idx, row in self.df.iterrows():
            project_name = row.get('Project')
            
            # Skip if no project name
            if pd.isna(project_name):
                continue
                
            print(f"Analyzing user growth for {project_name}...")
            
            # Calculate user growth score
            project_results = self.calculate_user_growth_score(idx)
            
            # Add to results
            results.append({
                'Project': project_name,
                'Market Sector': project_results['sector'],
                'User Growth Score': project_results['overall_score'],
                'Growth Category': project_results.get('growth_category'),
                'Details': project_results
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            # Save only the main columns, not the details
            columns_to_save = ['Project', 'Market Sector', 'User Growth Score']
            
            # Only include Growth Category if it exists in the DataFrame
            if 'Growth Category' in results_df.columns:
                columns_to_save.append('Growth Category')
                
            # Filter to only columns that exist
            save_cols = [col for col in columns_to_save if col in results_df.columns]
            
            # Save the results
            save_df = results_df[save_cols]
            save_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df
    
    def visualize_sector_growth(self, output_file: Optional[str] = None) -> None:
        """
        Create visualizations of user growth by sector.
        
        Args:
            output_file: Optional path to save visualization
        """
        # First, analyze all projects
        results_df = self.analyze_all_projects()
        
        if results_df.empty:
            print("No results to visualize")
            return
            
        # Set up the visualization
        plt.figure(figsize=(14, 10))
        sns.set(style="whitegrid")
        
        # Filter out rows with null scores
        results_df = results_df.dropna(subset=['User Growth Score'])
        
        # Plot 1: Box plot of scores by sector
        plt.subplot(2, 1, 1)
        sector_data = results_df.groupby('Market Sector')['User Growth Score'].agg(['mean', 'count'])
        sector_data = sector_data.sort_values('mean', ascending=False)
        
        # Only show sectors with at least 3 projects
        sectors_to_plot = sector_data[sector_data['count'] >= 3].index.tolist()
        
        if sectors_to_plot:
            plot_df = results_df[results_df['Market Sector'].isin(sectors_to_plot)]
            sns.boxplot(x='Market Sector', y='User Growth Score', data=plot_df, 
                        order=sectors_to_plot)
            plt.xticks(rotation=45, ha='right')
            plt.title('User Growth Scores by Market Sector')
            plt.tight_layout()
        
        # Plot 2: Top 20 projects by growth score
        plt.subplot(2, 1, 2)
        top_projects = results_df.nlargest(20, 'User Growth Score')
        
        sns.barplot(x='User Growth Score', y='Project', data=top_projects, 
                    hue='Market Sector', dodge=False)
        plt.title('Top 20 Projects by User Growth Score')
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
            
        plt.show()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with crypto project data
    """
    # Determine file type from extension
    if file_path.lower().endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            try:
                # Try again with more options
                df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
            except:
                try:
                    # Last attempt with minimal parsing
                    df = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
                except Exception as e2:
                    raise ValueError(f"Could not read CSV file: {e2}")
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        try:
            # Try loading with default settings
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            try:
                # Try with multi-level headers
                df = pd.read_excel(file_path, header=[0, 1])
                print("Loaded Excel with multi-level headers")
                
                # Flatten the column names
                df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            except Exception as e2:
                raise ValueError(f"Could not read Excel file: {e2}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def main():
    """Main function to run the user growth analysis."""
    parser = argparse.ArgumentParser(description='Analyze user growth for crypto projects')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to data file (CSV or Excel)')
    parser.add_argument('--output', type=str, default='user_growth_results.csv',
                        help='Path to output CSV file')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--viz-output', type=str,
                        help='Path to save visualization (PNG)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data(args.data)
        
        # Show columns for debugging
        if args.debug:
            print("\nDataFrame Columns:")
            for i, col in enumerate(df.columns):
                print(f"{i}: {col}")
            
            # Check if we have key columns
            required_cols = ['Project', 'Market sector']
            for col in required_cols:
                if col in df.columns:
                    print(f"Found required column: {col}")
                    print(f"Sample values: {df[col].dropna().head(3).tolist()}")
                else:
                    print(f"Missing required column: {col}")
                    
                    # Try to find similar columns
                    similar = [c for c in df.columns if col.lower() in str(c).lower()]
                    if similar:
                        print(f"Similar columns found: {similar}")
        
        # Initialize the analyzer
        analyzer = UserGrowthAnalyzer(df)
        
        # Run analysis
        results = analyzer.analyze_all_projects(args.output)
        
        # Create visualizations if requested
        if args.visualize:
            analyzer.visualize_sector_growth(args.viz_output)
        
        print(f"Analysis complete. Found {len(results)} projects with growth data.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()