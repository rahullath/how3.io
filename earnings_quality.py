import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import sys

def load_excel(file_path: str) -> pd.DataFrame:
    """
    Load an Excel file with multi-level headers and fix unnamed headers.

    Args:
        file_path: Path to the Excel file.

    Returns:
        A pandas DataFrame with flattened and fixed column names.
    """
    # Load the Excel file with multi-level headers
    df = pd.read_excel(file_path, header=[0, 1])

    # Fix unnamed headers explicitly
    new_columns = []
    for top, bottom in df.columns:
        if 'Unnamed' in top:
            new_columns.append(bottom.strip())
        else:
            new_columns.append(f"{top.strip()}_{bottom.strip()}")
    df.columns = new_columns

    return df

class EarningsQualityAnalyzer:
    """
    A specialized class for analyzing the quality and sustainability of earnings/revenue
    for cryptocurrency projects.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe containing crypto project metrics."""
        self.df = df
        self._fix_column_names()
        self.revenue_columns = self._find_revenue_columns()
        self.stability_columns = self._find_stability_columns()
        self.diversification_columns = self._find_diversification_columns()
    
    def _fix_column_names(self):
        """Fix common issues with column names."""
        # Check if we have Project and Market sector columns
        has_project = False
        has_market_sector = False
        
        # Look for Project column
        for col in self.df.columns:
            if col == 'Project' or (isinstance(col, str) and col.lower() == 'project'):
                has_project = True
                if col != 'Project':
                    self.df.rename(columns={col: 'Project'}, inplace=True)
                    print(f"Renamed '{col}' to 'Project'")
                break
        
        # Look for Market sector column
        for col in self.df.columns:
            if col == 'Market sector' or (isinstance(col, str) and col.lower() in ['market sector', 'marketsector', 'sector']):
                has_market_sector = True
                if col != 'Market sector':
                    self.df.rename(columns={col: 'Market sector'}, inplace=True)
                    print(f"Renamed '{col}' to 'Market sector'")
                break
        
        # If we still don't have Project column, look for unnamed columns
        if not has_project or not has_market_sector:
            # Check if there are unnamed columns that might contain our data
            unnamed_cols = [col for col in self.df.columns if 'Unnamed:' in str(col)]
            
            if len(unnamed_cols) >= 2:
                print(f"Found {len(unnamed_cols)} unnamed columns. Checking if they contain project and sector data...")
                
                # Check if the first row might contain headers
                if len(self.df) > 0:
                    first_row = self.df.iloc[0]
                    found_project = False
                    found_sector = False
                    
                    for i, col in enumerate(unnamed_cols):
                        val = first_row[col]
                        if isinstance(val, str):
                            if val.lower() in ['project', 'name', 'token', 'cryptocurrency']:
                                # This might be the project column
                                self.df.rename(columns={col: 'Project'}, inplace=True)
                                print(f"Renamed '{col}' to 'Project' based on first row value")
                                found_project = True
                            elif 'sector' in val.lower() or 'category' in val.lower():
                                # This might be the market sector column
                                self.df.rename(columns={col: 'Market sector'}, inplace=True)
                                print(f"Renamed '{col}' to 'Market sector' based on first row value")
                                found_sector = True
                    
                    # If we found headers in first row, drop it
                    if found_project or found_sector:
                        print("First row appears to contain headers. Dropping it...")
                        self.df = self.df.iloc[1:].reset_index(drop=True)
                
                # If we still don't have Project column, use the first unnamed column
                if 'Project' not in self.df.columns and len(unnamed_cols) > 0:
                    self.df.rename(columns={unnamed_cols[0]: 'Project'}, inplace=True)
                    print(f"Using first unnamed column '{unnamed_cols[0]}' as 'Project'")
                
                # If we still don't have Market sector column, use the second unnamed column
                if 'Market sector' not in self.df.columns and len(unnamed_cols) > 1:
                    self.df.rename(columns={unnamed_cols[1]: 'Market sector'}, inplace=True)
                    print(f"Using second unnamed column '{unnamed_cols[1]}' as 'Market sector'")
        
        # Check if we have data in these columns
        if 'Project' in self.df.columns:
            # If all values are null, try to find another column
            if self.df['Project'].isna().all():
                print("Project column contains all null values. Looking for alternative...")
                
                # Try to find a column with project names
                for col in self.df.columns:
                    if col != 'Project' and col != 'Market sector':
                        non_null_vals = self.df[col].dropna()
                        if len(non_null_vals) > 0 and all(isinstance(x, str) for x in non_null_vals):
                            self.df['Project'] = self.df[col]
                            print(f"Using '{col}' values for Project")
                            break
        
        # Same for Market sector
        if 'Market sector' in self.df.columns:
            # If all values are null, try to find another column
            if self.df['Market sector'].isna().all():
                print("Market sector column contains all null values. Looking for alternative...")
                
                # Try to find a column with sector names
                for col in self.df.columns:
                    if col != 'Project' and col != 'Market sector':
                        non_null_vals = self.df[col].dropna()
                        if len(non_null_vals) > 0 and all(isinstance(x, str) for x in non_null_vals):
                            if any('blockchain' in str(x).lower() or 'defi' in str(x).lower() for x in non_null_vals):
                                self.df['Market sector'] = self.df[col]
                                print(f"Using '{col}' values for Market sector")
                                break
        
        # Print the results
        if 'Project' in self.df.columns:
            non_null_projects = self.df['Project'].dropna()
            print(f"Project column has {len(non_null_projects)} non-null values")
            if len(non_null_projects) > 0:
                print(f"Sample projects: {non_null_projects.iloc[:5].tolist()}")
        
        if 'Market sector' in self.df.columns:
            non_null_sectors = self.df['Market sector'].dropna()
            print(f"Market sector column has {len(non_null_sectors)} non-null values")
            if len(non_null_sectors) > 0:
                print(f"Sample sectors: {non_null_sectors.unique()[:5].tolist()}")
    
    def _find_revenue_columns(self) -> Dict[str, List[str]]:
        """
        Find columns related to revenue and fee metrics.

        Returns:
            A dictionary with primary, secondary, and tertiary revenue columns.
        """
        revenue_cols = {
            'primary': [],
            'secondary': [],
            'tertiary': []
        }

        # Primary revenue/fee columns - these are the main metrics
        primary_keywords = ['revenue', 'fees', 'earnings', 'protocol fees']

        # Secondary columns - alternative revenue metrics
        secondary_keywords = ['supply-side fees', 'transaction fees', 'trading fees', 'average fee per user']

        # Tertiary columns - related metrics that could be used if others aren't available
        tertiary_keywords = ['average transaction fee', 'fee per transaction', 'gross profit', 'p/s ratio']

        # Check all columns
        for col in self.df.columns:
            col_str = str(col).lower()

            # Check primary metrics first
            if any(keyword in col_str for keyword in primary_keywords):
                if any(period in col_str for period in ['24h', '7d', '30d', '90d', '180d', '365d']):
                    revenue_cols['primary'].append(col)

            # Check secondary metrics
            elif any(keyword in col_str for keyword in secondary_keywords):
                if any(period in col_str for period in ['24h', '7d', '30d', '90d', '180d', '365d']):
                    revenue_cols['secondary'].append(col)

            # Check tertiary metrics
            elif any(keyword in col_str for keyword in tertiary_keywords):
                if any(period in col_str for period in ['24h', '7d', '30d', '90d', '180d', '365d']):
                    revenue_cols['tertiary'].append(col)

        # Sort columns by time period to help with stability calculations
        for category in revenue_cols:
            revenue_cols[category] = sorted(revenue_cols[category], key=lambda x: self._get_time_period_order(x))

        # Print found columns
        for category, cols in revenue_cols.items():
            print(f"Found {len(cols)} {category} revenue columns")
            if cols:
                print(f"Sample columns: {cols[:3]}")

        return revenue_cols
    
    def _get_time_period_order(self, col_name: str) -> int:
        """
        Return a sortable value based on the time period in the column name.

        Args:
            col_name: The name of the column.

        Returns:
            An integer representing the order of the time period.
        """
        # Define a priority for sorting columns
        periods = {
            '24h': 1,
            '7d': 2,
            '30d': 3,
            '90d': 4,
            '180d': 5,
            '365d': 6
        }
        for period, order in periods.items():
            if period in col_name:
                return order
        # Default high number if no known period is found
        return 999
    
    def _find_stability_columns(self) -> List[str]:
        """Find columns that can be used to measure revenue stability."""
        stability_cols = []
        
        # Look for columns with change/trend information
        stability_keywords = [
            'change', 'trend', 'growth', 'volatility', 'stability'
        ]
        
        time_periods = ['30d', '90d', '180d', '365d']
        
        # Find revenue/fee columns with change metrics
        for col in self.df.columns:
            col_str = str(col).lower()
            
            # Check if column contains revenue/fee and change/trend
            if any(rev in col_str for rev in ['revenue', 'fees', 'earnings']):
                if any(stab in col_str for stab in stability_keywords):
                    if any(period in col_str for period in time_periods):
                        stability_cols.append(col)
        
        print(f"Found {len(stability_cols)} stability columns")
        if stability_cols:
            print(f"Sample columns: {stability_cols[:3]}")
        
        return stability_cols
    
    def _find_diversification_columns(self) -> List[str]:
        """Find columns related to revenue diversification."""
        diversification_cols = []
        
        # Look for columns with revenue breakdown information
        diversification_keywords = [
            'breakdown', 'source', 'category', 'segment', 'diversification',
            'cost of revenue', 'operating expenses', 'gross profit'
        ]
        
        for col in self.df.columns:
            col_str = str(col).lower()
            
            # Check for revenue diversification metrics
            if any(rev in col_str for rev in ['revenue', 'fees', 'earnings']):
                if any(div in col_str for div in diversification_keywords):
                    diversification_cols.append(col)
        
        print(f"Found {len(diversification_cols)} diversification columns")
        if diversification_cols:
            print(f"Sample columns: {diversification_cols[:3]}")
        
        return diversification_cols
    
    def calculate_quarterly_values(self, row: pd.Series) -> Dict[str, float]:
        """Calculate quarterly revenue/fee values for a project."""
        quarterly_values = {}

        # Try to find best quarterly data
        quarterly_cols = [col for col in self.revenue_columns['primary'] if '90d' in str(col).lower()]

        if quarterly_cols:
            for i, col in enumerate(quarterly_cols[:4]):  # Use up to 4 quarters
                value = self._get_numeric_value(row, col)
                if value is not None:
                    quarterly_values[f'Q{i+1}'] = value

        # If we don't have 4 quarters, try to derive from yearly data
        if len(quarterly_values) < 4:
            yearly_cols = [col for col in self.revenue_columns['primary'] if '365d' in str(col).lower()]
            if yearly_cols:
                yearly_col = yearly_cols[0]
                yearly_value = self._get_numeric_value(row, yearly_col)
                if yearly_value is not None:
                    missing_quarters = 4 - len(quarterly_values)
                    for i in range(missing_quarters):
                        q_num = len(quarterly_values) + 1
                        quarterly_values[f'Q{q_num}'] = yearly_value / 4

        return quarterly_values
    
    def _get_numeric_value(self, row: pd.Series, col_name: str) -> Optional[float]:
        """Extract a numeric value from the row."""
        value = row.get(col_name, np.nan)

        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            except (ValueError, TypeError):
                return None

        return None
    
    def calculate_revenue_stability(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate revenue stability score (0-100) based on quarter-over-quarter volatility.

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (stability_score, explanation_dict).
        """
        explanation = {
            'quarterly_values': {},
            'quarterly_changes': {},
            'volatility_measure': None,
            'has_high_volatility': False,
            'stability_score': None,
            'data_completeness': 0.0,
            'method_used': 'No valid data found'
        }

        # Example logic for stability calculation
        quarterly_values = self.calculate_quarterly_values(row)
        explanation['quarterly_values'] = quarterly_values

        if len(quarterly_values) < 2:
            explanation['method_used'] = 'Insufficient quarterly data'
            return None, explanation

        # Calculate quarter-over-quarter changes
        quarters = sorted(quarterly_values.keys())
        qoq_changes = []
        for i in range(1, len(quarters)):
            current = quarterly_values[quarters[i]]
            previous = quarterly_values[quarters[i - 1]]
            if previous != 0:
                change_pct = abs((current - previous) / previous) * 100
                qoq_changes.append(change_pct)
                explanation['quarterly_changes'][f"{quarters[i-1]} to {quarters[i]}"] = change_pct

        if not qoq_changes:
            explanation['method_used'] = 'Could not calculate quarterly changes'
            return None, explanation

        # Average volatility
        avg_volatility = sum(qoq_changes) / len(qoq_changes)
        explanation['volatility_measure'] = avg_volatility
        explanation['data_completeness'] = len(quarterly_values) / 4.0

        # Convert to stability score (inverse of volatility)
        stability_score = 100 - min(avg_volatility * 2, 100)
        explanation['stability_score'] = stability_score
        explanation['method_used'] = 'Calculated from quarterly revenue changes'

        return stability_score, explanation
    
    def calculate_revenue_diversification(self, row: pd.Series) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate revenue diversification score (0-100) based on distribution of revenue sources.

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (diversification_score, explanation_dict).
        """
        explanation = {
            'revenue_sources': {},
            'largest_source_pct': None,
            'num_significant_sources': 0,
            'diversification_score': None,
            'method_used': 'No valid data found'
        }

        # Method 1: Use direct diversification columns if available
        if self.diversification_columns:
            # Extract revenue breakdown by source
            source_values = {}
            for col in self.diversification_columns:
                value = self._get_numeric_value(row, col)
                if value is not None and value > 0:
                    source_name = str(col).split('_')[-1] if '_' in str(col) else col
                    source_values[source_name] = value

            if source_values:
                explanation['revenue_sources'] = source_values
                explanation['method_used'] = 'Direct measurement from revenue source columns'

                # Calculate largest source percentage
                total_revenue = sum(source_values.values())
                largest_source = max(source_values.values())
                largest_pct = (largest_source / total_revenue) * 100 if total_revenue > 0 else 100

                explanation['largest_source_pct'] = largest_pct

                # Count significant sources (>5% of total)
                significant_sources = sum(1 for v in source_values.values() if v / total_revenue > 0.05)
                explanation['num_significant_sources'] = significant_sources

                # Calculate diversification score
                theoretical_even_pct = 100 / len(source_values)
                diversification_score = 100 - (largest_pct - theoretical_even_pct)
                diversification_score = max(0, min(100, diversification_score))

                explanation['diversification_score'] = diversification_score
                return diversification_score, explanation

        # No direct data available
        return None, explanation
    
    def calculate_earnings_quality(self, row: pd.Series, min_projects_for_percentile: int = 3) -> Dict[str, Any]:
        """
        Calculate overall earnings quality score combining stability and diversification.

        Args:
            row: A pandas Series representing a single row of the DataFrame.
            min_projects_for_percentile: Minimum number of projects needed in sector for percentile calculations.

        Returns:
            Dictionary with earnings quality scores and explanations.
        """
        project_val = row.get('Project', "Unknown")
        market_sector = row.get('Market sector', "Unknown")

        result = {
            'stability': {
                'score': None,
                'percentile': None,
                'explanation': None
            },
            'diversification': {
                'score': None,
                'percentile': None,
                'explanation': None
            },
            'overall_score': None,
            'metrics_used': [],
            'complete_score': False
        }

        # Stability calculation
        stability_score, stability_explanation = self.calculate_revenue_stability(row)
        result['stability']['score'] = stability_score
        result['stability']['explanation'] = stability_explanation

        if stability_score is not None:
            result['metrics_used'].append('stability')

        # Diversification calculation
        diversification_score, diversification_explanation = self.calculate_revenue_diversification(row)
        result['diversification']['score'] = diversification_score
        result['diversification']['explanation'] = diversification_explanation

        if diversification_score is not None:
            result['metrics_used'].append('diversification')

        # Calculate overall earnings quality score
        valid_percentiles = []
        if stability_score is not None:
            valid_percentiles.append(stability_score)
        if diversification_score is not None:
            valid_percentiles.append(diversification_score)

        if valid_percentiles:
            # Weight stability higher than diversification if both are available
            if len(valid_percentiles) == 2:
                result['overall_score'] = 0.7 * stability_score + 0.3 * diversification_score
                result['complete_score'] = True
            else:
                result['overall_score'] = valid_percentiles[0]
                result['complete_score'] = len(valid_percentiles) == 2

        return result


def enhance_earnings_quality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance earnings quality analysis for all projects in the dataset.
    
    Args:
        df: DataFrame with crypto project metrics
        
    Returns:
        DataFrame with enhanced earnings quality scores
    """
    # Create analyzer
    analyzer = EarningsQualityAnalyzer(df)
    
    # Create result columns
    results = []
    
    # Process each project
    for idx, row in df.iterrows():
        project_name = row.get('Project', "Unknown")
        market_sector = row.get('Market sector', "Unknown")

        if pd.isna(project_name) or project_name.lower() in ['project', 'name', '#', 'token', 'unknown']:
            continue

        print(f"Processing {project_name} ({market_sector})...")

        quality_results = analyzer.calculate_earnings_quality(row, min_projects_for_percentile=3)

        result = {
            'Project': project_name,
            'Market Sector': market_sector,
            'Revenue Stability Score': quality_results['stability']['score'],
            'Revenue Diversification Score': quality_results['diversification']['score'],
            'Earnings Quality Score': quality_results['overall_score'],
            'Metrics Used': ', '.join(quality_results['metrics_used']),
            'Complete Score': quality_results['complete_score']
        }

        results.append(result)

    results_df = pd.DataFrame(results)
    
    return results_df


def visualize_earnings_quality(results_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Create visualizations for earnings quality analysis.
    
    Args:
        results_df: DataFrame with earnings quality results
        top_n: Number of top projects to show in visualizations
    """
    if results_df.empty:
        print("No results to visualize.")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Top projects by Earnings Quality Score
    if 'Earnings Quality Score' in results_df.columns:
        top_projects = results_df.dropna(subset=['Earnings Quality Score']).sort_values('Earnings Quality Score', ascending=False).head(top_n)
        
        if not top_projects.empty:
            ax = axes[0, 0]
            sns.barplot(x='Earnings Quality Score', y='Project', data=top_projects, ax=ax)
            ax.set_title(f'Top {top_n} Projects by Earnings Quality')
            ax.set_xlim(0, 100)
    
    # 2. Distribution of Stability Scores
    if 'Revenue Stability Score' in results_df.columns:
        stability_scores = results_df['Revenue Stability Score'].dropna()
        if not stability_scores.empty:
            ax = axes[0, 1]
            sns.histplot(stability_scores, kde=True, ax=ax)
            ax.set_title('Distribution of Revenue Stability Scores')
            ax.set_xlim(0, 100)
    
    # 3. Scatter plot of Stability vs Diversification
    if 'Revenue Stability Score' in results_df.columns and 'Revenue Diversification Score' in results_df.columns:
        scatter_data = results_df.dropna(subset=['Revenue Stability Score', 'Revenue Diversification Score'])
        if not scatter_data.empty:
            ax = axes[1, 0]
            
            # Check if we have enough unique sectors for hue
            if 'Market Sector' in scatter_data.columns and scatter_data['Market Sector'].nunique() > 1:
                sns.scatterplot(
                    x='Revenue Stability Score', 
                    y='Revenue Diversification Score', 
                    hue='Market Sector',
                    size='Earnings Quality Score',
                    sizes=(20, 200),
                    alpha=0.7,
                    data=scatter_data,
                    ax=ax
                )
            else:
                # If no sector information, use simpler plot
                sns.scatterplot(
                    x='Revenue Stability Score', 
                    y='Revenue Diversification Score', 
                    size='Earnings Quality Score',
                    sizes=(20, 200),
                    alpha=0.7,
                    data=scatter_data,
                    ax=ax
                )
            
            ax.set_title('Revenue Stability vs Diversification')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            # Add legend if we have multiple sectors
            if 'Market Sector' in scatter_data.columns and scatter_data['Market Sector'].nunique() > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Average Earnings Quality by Market Sector
    if 'Earnings Quality Score' in results_df.columns and 'Market Sector' in results_df.columns:
        # Only proceed if we have multiple sectors
        if results_df['Market Sector'].nunique() > 1:
            ax = axes[1, 1]
            
            # Calculate average scores by sector
            sector_scores = results_df.groupby('Market Sector')['Earnings Quality Score'].mean().sort_values(ascending=False)
            sector_counts = results_df.groupby('Market Sector').size()
            
            # Filter to sectors with enough data
            valid_sectors = sector_scores[sector_counts >= 3].index
            filtered_sector_scores = sector_scores[valid_sectors]
            
            # Plot if we have data
            if not filtered_sector_scores.empty:
                sns.barplot(x=filtered_sector_scores.values, y=filtered_sector_scores.index, ax=ax)
                ax.set_title('Average Earnings Quality by Market Sector')
                ax.set_xlim(0, 100)
            else:
                ax.text(0.5, 0.5, 'Not enough data per sector\nfor meaningful comparison', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            # No multiple sectors
            ax = axes[1, 1]
            ax.text(0.5, 0.5, 'Only one market sector found\nCannot compare across sectors', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('earnings_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def debug_data_structure(file_path: str) -> None:
    """Debug the data structure of the file to assist with parsing."""
    print(f"\nDEBUGGING FILE: {file_path}")
    
    try:
        # Determine file type
        if file_path.lower().endswith('.csv'):
            # For CSV, read the first few rows and report columns
            df = pd.read_csv(file_path, nrows=5)
            print(f"CSV file with {df.shape[1]} columns")
            print(f"Column names: {df.columns.tolist()[:10]}...")
            
            # Check for unnamed columns
            unnamed = [col for col in df.columns if 'Unnamed:' in str(col)]
            if unnamed:
                print(f"Found {len(unnamed)} unnamed columns")
                
                # Check first row as potential headers
                first_row = df.iloc[0]
                print(f"First row values: {first_row.tolist()[:10]}...")
                
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            # For Excel, check sheet names and first few rows
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            print(f"Excel file with sheets: {wb.sheetnames}")
            
            # Get first sheet
            sheet = wb[wb.sheetnames[0]]
            print(f"First sheet: {sheet.title}")
            
            # Report dimensions
            print(f"Sheet dimensions: {sheet.max_row} rows, {sheet.max_column} columns")
            
            # Check first few cells
            first_row = []
            for cell in list(sheet.iter_rows(min_row=1, max_row=1, values_only=True))[0]:
                first_row.append(cell)
            print(f"First row values: {first_row[:10]}...")
            
            # Check second row
            second_row = []
            for cell in list(sheet.iter_rows(min_row=2, max_row=2, values_only=True))[0]:
                second_row.append(cell)
            print(f"Second row values: {second_row[:10]}...")
            
        else:
            print(f"Unsupported file format: {file_path}")
    
    except Exception as e:
        print(f"Error debugging file: {e}")


# Example usage
file_path = "tt-master-data_2025-03-27.xlsx"
df = load_excel(file_path)

# Verify the flattened column names
print("Flattened Columns:")
print(df.columns.tolist())

if __name__ == "__main__":
    import sys
    import os
    
    # Check for debugging flag
    debug_mode = '--debug' in sys.argv
    if debug_mode:
        sys.argv.remove('--debug')
    
    # Find input file
    data_file = None
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Try common filenames
        for filename in ['tt-master-data_2025-03-27.xlsx', 'ttmasterdata_20250327.xlsx', 'crypto_data.xlsx']:
            if os.path.exists(filename):
                data_file = filename
                break
    
    if not data_file or not os.path.exists(data_file):
        print("Error: Data file not found")
        print("Usage: python earnings_quality.py [data_file.xlsx] [--debug]")
        sys.exit(1)
    
    # Run in debug mode if requested
    if debug_mode:
        debug_data_structure(data_file)
        sys.exit(0)
    
    print(f"Loading data from {data_file}...")
    
    # Load data
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = load_excel(data_file)
        
        print(f"Loaded data with {len(df)} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run enhanced analysis
    results = enhance_earnings_quality_analysis(df)
    
    # Check if we got results
    if results.empty:
        print("No valid results were generated. Try running with --debug to diagnose the issue.")
        sys.exit(1)
    
    # Save results
    output_file = 'earnings_quality_results.csv'
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Create visualizations
    visualize_earnings_quality(results)
    print("Visualizations saved to earnings_quality_analysis.png")