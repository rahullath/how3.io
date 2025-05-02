import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
from earnings_quality_results_exporter import export_results_to_csv

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
        if 'Unnamed' in str(top):
            new_columns.append(str(bottom).strip())
        else:
            new_columns.append(f"{str(top).strip()}_{str(bottom).strip()}")
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
        

        # Find all columns related to revenue metrics
        all_columns = self.df.columns.tolist()
        self.revenue_columns = self._find_revenue_columns(all_columns)
        self.stability_columns = self._find_stability_columns(all_columns)
        self.diversification_columns = self._find_diversification_columns(all_columns)
        
        # Track columns to use for calculations
        self.fee_columns = []
        self.trend_columns = []
        self.user_columns = []
        self.treasury_columns = []
        self.developer_columns = []
        self.transaction_columns = []
        
        self._find_metric_columns(all_columns)
        
        # Print summary of found columns
        print(f"Found {len(self.revenue_columns['primary'])} primary revenue columns")
        print(f"Found {len(self.stability_columns)} stability columns")
        print(f"Found {len(self.diversification_columns)} diversification columns")
        print(f"Found {len(self.fee_columns)} fee columns")
        print(f"Found {len(self.trend_columns)} trend columns")
        print(f"Found {len(self.user_columns)} user columns")
        print(f"Found {len(self.treasury_columns)} treasury columns")
        print(f"Found {len(self.developer_columns)} developer columns")
        print(f"Found {len(self.transaction_columns)} transaction columns")
        
        #Calculate comprehensive score and save the explanations
        
    
    def _fix_column_names(self):
        """Fix common issues with column names."""
        # Check if we have Project and Market sector columns
        has_project = False
        has_market_sector = False
        
        print("Columns in DataFrame:", self.df.columns.tolist())
        # Look for Project column
        print("Columns in DataFrame:", self.df.columns.tolist())
        if len(self.df) > 0:
            print("First row values:", self.df.iloc[0].tolist()[:10])

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
    
    def _find_revenue_columns(self, all_columns: List[str]) -> Dict[str, List[str]]:
        """
        Identify columns related to revenue metrics.

        Returns:
            Dict with categorized revenue columns
        """
        revenue_cols = {
            'primary': [],
            'fee': [],
            'trend': [],
            'stability': []
        }
        
        # Revenue/fee primary keywords
        primary_keywords = [
            'revenue', 'fees', 'earning', 'profit', 'income',
            'supply-side fee', 'transaction fee', 'protocol revenue', 'yield'
        ]
        
        # Look for primary revenue columns
        for col in all_columns:
            col_lower = col.lower()
            
            # Primary revenue columns (raw values)
            if any(keyword in col_lower for keyword in primary_keywords):
                revenue_cols['primary'].append(col)
                    
                # Fee specific columns
                if any(fee in col_lower for fee in ['fee', 'fees']):
                    revenue_cols['fee'].append(col)
                    
                # Trend specific columns
                if any(trend in col_lower for trend in ['trend', 'change', 'growth']):
                    revenue_cols['trend'].append(col)
                    
        # Stability-specific keywords and periods for advanced stability calculations
        stability_keywords = ['fees', 'supply-side fees', 'earnings', 'revenue', 'trend', 'change', 'growth', 'apy']
        stability_periods = ['30d trend', '90d trend', '180d trend', '365d trend']

        for col in all_columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in stability_keywords) \
                    and any(period in col_lower for period in stability_periods):
                revenue_cols['stability'].append(col)

        # Print found columns for debugging
        for category, cols in revenue_cols.items():
            if cols:
                print(f"Found {len(cols)} {category} revenue columns")
                print(f"Sample columns: {cols[:min(3, len(cols))]}")

        return revenue_cols
    
    def _find_metric_columns(self, all_columns: List[str]) -> None:
        """Find specific metric columns for calculations."""
        # Maps of keywords to column lists
        metric_mappings = {
            'fee': ['fee', 'fees', 'revenue', 'earning'],
            'trend': ['trend', 'change', 'growth', 'volatility'],
            'user': ['user', 'arpu', 'average revenue per user', 'active user', 'dau', 'mau'],
            'treasury': ['treasury', 'fund', 'reserve'],
            'developer': ['developer', 'engineer', 'contributor', 'core dev'],
            'transaction': ['transaction', 'tx', 'volume', 'trade']
        }
        
        # Check each column against each metric category
        for col in all_columns:
            col_lower = col.lower()
            
            # Check for fee columns
            if any(keyword in col_lower for keyword in metric_mappings['fee']):
                if 'sum' in col_lower or 'latest' in col_lower:
                    self.fee_columns.append(col)
            
            # Check for trend columns 
            if any(keyword in col_lower for keyword in metric_mappings['trend']):
                if any(period in col_lower for period in ['30d', '90d', '180d', '365d']):
                    self.trend_columns.append(col)
            
            # Check for user metrics
            if any(keyword in col_lower for keyword in metric_mappings['user']):
                self.user_columns.append(col)
            
            # Check for treasury metrics
            if any(keyword in col_lower for keyword in metric_mappings['treasury']):
                if 'latest' in col_lower:
                    self.treasury_columns.append(col)
            
            # Check for developer metrics
            if any(keyword in col_lower for keyword in metric_mappings['developer']):
                if 'latest' in col_lower or 'count' in col_lower:
                    self.developer_columns.append(col)
            
            # Check for transaction metrics
            if any(keyword in col_lower for keyword in metric_mappings['transaction']):
                if 'count' in col_lower or 'latest' in col_lower or 'sum' in col_lower:
                    self.transaction_columns.append(col)
    
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
    
    def _find_stability_columns(self, all_columns: List[str]) -> List[str]:
        """Find columns that can be used to measure revenue stability."""
        stability_cols = []
        
        # Look for columns with change/trend information
        stability_keywords = [
            'change', 'trend', 'growth', 'volatility', 'stability'
        ]
        
        time_periods = ['30d', '90d', '180d', '365d']
        
        # Find revenue/fee columns with change metrics
        for col in all_columns:
            col_str = str(col).lower()
            
            # Check if column contains revenue/fee and change/trend
            if any(rev in col_str for rev in ['revenue', 'fees', 'earnings']):
                if any(stab in col_str for stab in stability_keywords):
                    if any(period in col_str for period in time_periods):
                        stability_cols.append(col)
        
        print(f"Found {len(stability_cols)} stability columns")
        if stability_cols:
            print(f"Sample columns: {stability_cols[:min(3, len(stability_cols))]}")
        
        return stability_cols
    
    def _find_diversification_columns(self, all_columns: List[str]) -> List[str]:
        """Find columns related to revenue diversification."""
        diversification_cols = []
        
        # Look for columns with revenue breakdown information
        diversification_keywords = [
            'breakdown', 'source', 'category', 'segment', 'diversification',
            'cost of revenue', 'operating expenses', 'gross profit'
        ]
        
        for col in all_columns:
            col_str = str(col).lower()
            
            # Check for revenue diversification metrics
            if any(rev in col_str for rev in ['revenue', 'fees', 'earnings']):
                if any(div in col_str for div in diversification_keywords):
                    diversification_cols.append(col)
        
        print(f"Found {len(diversification_cols)} diversification columns")
        if diversification_cols:
            print(f"Sample columns: {diversification_cols[:min(3, len(diversification_cols))]}")
        
        return diversification_cols
    
    def _get_numeric_value(self, value) -> Optional[float]:
        """Extract a numeric value from a cell value."""
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove currency symbols and commas
            for char in ['$', '€', '£', ',', '%']:
                value = value.replace(char, '')
            
            # Try to convert to float
            try:
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            except (ValueError, TypeError):
                return None

        return None
        
    def calculate_stability_score(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate revenue stability score (0-100) based on prioritized metrics.

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (stability_score, explanation_dict).
        """
        explanation = {
            "chosen_metric": None,
            "metric_values": {},
            "trends": {},
            "stability_score": None,
            "method": "No sufficient data"
        }
        
        # First try to find paired revenue/trend data
        best_revenue_col = None
        best_trend_col = None
        
        # Look for best revenue column (30d sum preferred)
        for col in self.fee_columns:
            col_lower = col.lower()
            if '30d' in col_lower and 'sum' in col_lower:
                value = self._get_numeric_value(row.get(col))
                if value is not None and value > 0:
                    best_revenue_col = col
                    break
        
        # If no 30d sum, try 7d or any other
        if not best_revenue_col:
            for col in self.fee_columns:
                col_lower = col.lower()
                if ('7d' in col_lower or 'latest' in col_lower) and 'sum' in col_lower:
                    value = self._get_numeric_value(row.get(col))
                    if value is not None and value > 0:
                        best_revenue_col = col
                        break
        
        # Now find a matching trend column
        if best_revenue_col:
            metric_base = best_revenue_col.split('_')[0]  # Get the metric type (e.g., 'Revenue')
            
            # Try to find matching trend
            for col in self.trend_columns:
                if metric_base in col and 'trend' in col.lower():
                    trend_value = self._get_numeric_value(row.get(col))
                    if trend_value is not None:
                        best_trend_col = col
                        break

        # If we found a revenue and a trend column, calculate stability score
        if best_revenue_col and best_trend_col:
            revenue_value = self._get_numeric_value(row.get(best_revenue_col))
            trend_value = self._get_numeric_value(row.get(best_trend_col))
            
            explanation["chosen_metric"] = best_revenue_col.split('_')[0]
            explanation["metric_values"] = {best_revenue_col: revenue_value}
            explanation["trends"] = {best_trend_col: trend_value}
            
            # Stability is inversely proportional to absolute trend value
            # A higher trend (positive or negative) means less stability
            abs_trend = abs(trend_value)
            
            # Normalize trend to a 0-100 range (0 = most stable, 100 = least stable)
            # Using a logistic function to handle outliers better
            normalized_trend = 100 / (1 + np.exp(-0.5 * (abs_trend - 20)))
            stability_score = 100 - normalized_trend
            
            explanation["stability_score"] = round(stability_score, 2)
            explanation["method"] = f"Calculated from {best_revenue_col} and {best_trend_col}"
            
            return stability_score, explanation
        
        # If we couldn't find paired data, try using just trend data
        elif self.trend_columns:
            # Use the average of available trend values
            trend_values = []
            for col in self.trend_columns:
                trend_value = self._get_numeric_value(row.get(col))
                if trend_value is not None:
                    trend_values.append(abs(trend_value))
                    explanation["trends"][col] = trend_value
            
            if trend_values:
                avg_trend = np.mean(trend_values)
                # Normalize trend using sigmoid function
                normalized_trend = 100 / (1 + np.exp(-0.5 * (avg_trend - 20)))
                stability_score = 100 - normalized_trend
                
                explanation["stability_score"] = round(stability_score, 2)
                explanation["method"] = "Calculated from average trend values"
                explanation["chosen_metric"] = "Average of trend metrics"
                
                return stability_score, explanation
        
        # Fallback to a default stability score with explanation
        explanation["method"] = "Insufficient stability data, using default"
        default_score = 30.0  # Assuming moderate stability without data
        return default_score, explanation
    
    def calculate_revenue_diversification(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate revenue diversification score (0-100).
        
        Args:
            row: A pandas Series representing a single row of the DataFrame.
            
        Returns:
            Tuple of (diversification_score, explanation_dict).
        """
        explanation = {
            'diversification_values': {},
            'diversification_score': None,
            'method_used': 'No valid data'
        }
        
        # Use any available diversification columns
        if self.diversification_columns:
            # Extract valid values from diversification columns
            valid_values = {}
            for col in self.diversification_columns:
                value = self._get_numeric_value(row.get(col))
                if value is not None:
                    valid_values[col] = value
            
            # If we have enough data, calculate diversification
            if len(valid_values) >= 2:
                # Calculate coefficient of variation (CV)
                values = list(valid_values.values())
                std_dev = np.std(values)
                mean_val = np.mean(values)
                
                if mean_val > 0:
                    # CV = std_dev / mean
                    cv = std_dev / mean_val
                    
                    # Map CV to a 0-100 scale (higher CV = higher diversification)
                    # Using a transforming function to get a reasonable range
                    diversification_score = min(100, 100 * cv / 2.0)
                    
                    explanation['diversification_values'] = valid_values
                    explanation['diversification_score'] = round(diversification_score, 2)
                    explanation['method_used'] = 'Calculated from diversification metrics'
                    
                    return diversification_score, explanation
        
        # If no diversification columns or insufficient data, use a fallback approach
        # Try to use fee columns that might represent different revenue streams
        revenue_cols = [col for col in self.fee_columns if '30d' in col.lower() or '7d' in col.lower()]
        
        if len(revenue_cols) >= 2:
            valid_values = {}
            for col in revenue_cols:
                value = self._get_numeric_value(row.get(col))
                if value is not None and value > 0:
                    valid_values[col] = value
            
            if len(valid_values) >= 2:
                values = list(valid_values.values())
                std_dev = np.std(values)
                mean_val = np.mean(values)
                
                if mean_val > 0:
                    cv = std_dev / mean_val
                    diversification_score = min(100, 100 * cv / 2.0)
                    
                    explanation['diversification_values'] = valid_values
                    explanation['diversification_score'] = round(diversification_score, 2)
                    explanation['method_used'] = 'Calculated from fee/revenue metrics'
                    
                    return diversification_score, explanation
        
        # Default to a moderate diversification score
        default_score = 40.0  # Middle-range default
        explanation['method_used'] = 'Insufficient diversification data, using default'
        explanation['diversification_score'] = default_score
        return default_score, explanation
    
    def calculate_user_efficiency(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate user efficiency score (0-100) based on ARPU (Average Revenue Per User).

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (user_efficiency_score, explanation_dict).
        """
        explanation = {
            "arpu_column": None,
            "user_column": None,
            "arpu_value": None,
            "percentile": None,
            "score": None,
            "method": "No valid data"
        }

        # Find ARPU and Active users columns
        arpu_col = None
        user_col = None
        
        # Look through user columns for ARPU
        for col in self.user_columns:
            col_lower = col.lower()
            if "arpu" in col_lower or "average revenue per user" in col_lower:
                arpu_col = col
                break
        
        # Look for user/active user columns
        for col in self.user_columns:
            col_lower = col.lower()
            if "active user" in col_lower or "dau" in col_lower or "mau" in col_lower:
                user_col = col
                break

        # If we don't have direct ARPU, try to calculate it from revenue and users
        if not arpu_col and user_col:
            # Try to find a revenue column to pair with user count
            for col in self.fee_columns:
                if '30d' in col.lower() and 'sum' in col.lower():
                    revenue_col = col
                    revenue = self._get_numeric_value(row.get(revenue_col))
                    users = self._get_numeric_value(row.get(user_col))
                    
                    if revenue is not None and users is not None and users > 0:
                        arpu = revenue / users
                        explanation["arpu_value"] = arpu
                        explanation["arpu_column"] = f"Calculated from {revenue_col} / {user_col}"
                        explanation["user_column"] = user_col
                        
                        # Compare to sector average
                        sector = row.get('Market sector', 'Unknown')
                        sector_rows = self.df[self.df['Market sector'] == sector]
                        
                        sector_arpus = []
                        for _, sector_row in sector_rows.iterrows():
                            s_revenue = self._get_numeric_value(sector_row.get(revenue_col))
                            s_users = self._get_numeric_value(sector_row.get(user_col))
                            if s_revenue is not None and s_users is not None and s_users > 0:
                                sector_arpus.append(s_revenue / s_users)
                        
                        if sector_arpus:
                            # Calculate percentile rank
                            percentile = sum(1 for x in sector_arpus if x <= arpu) / len(sector_arpus) * 100
                            explanation["percentile"] = percentile
                            
                            # Map percentile to 0-100 score (higher percentile = higher score)
                            score = min(100, percentile)
                            
                            explanation["score"] = round(score, 2)
                            explanation["method"] = f"Calculated from {revenue_col} / {user_col}"
                            return score, explanation
        
        # Direct ARPU approach
        if arpu_col:
            arpu = self._get_numeric_value(row.get(arpu_col))
            
            if arpu is not None and arpu > 0:
                explanation["arpu_value"] = arpu
                explanation["arpu_column"] = arpu_col
                
                # Compare to sector peers
                sector = row.get('Market sector', 'Unknown')
                sector_mask = self.df['Market sector'] == sector
                sector_projects = self.df[sector_mask]
                
                all_arpus = []
                for _, proj_row in sector_projects.iterrows():
                    proj_arpu = self._get_numeric_value(proj_row.get(arpu_col))
                    if proj_arpu is not None and proj_arpu > 0:
                        all_arpus.append(proj_arpu)
                
                if all_arpus:
                    # Calculate percentile (higher ARPU is better)
                    percentile = sum(1 for v in all_arpus if v <= arpu) / len(all_arpus) * 100
                    explanation["percentile"] = percentile
                    
                    # Convert to score (0-100)
                    score = min(100, percentile)
                    
                    explanation["score"] = round(score, 2)
                    explanation["method"] = f"Calculated from {arpu_col}"
                    return score, explanation
        
        # Default to a moderate efficiency score if no data available
        default_score = 35.0
        explanation["method"] = "Insufficient user efficiency data, using default"
        explanation["score"] = default_score
        return default_score, explanation
    
    def calculate_sustainability(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate sustainability score (0-100) based on Treasury and Core developers.

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (sustainability_score, explanation_dict).
        """
        explanation = {
            "treasury_column": None,
            "developers_column": None,
            "treasury_value": None,
            "developers_value": None,
            "score": None,
            "method": "No valid data"
        }

        # Find treasury and developer columns
        treasury_col = None
        developer_col = None

        # Look for treasury columns
        for col in self.treasury_columns:
            col_lower = col.lower()
            if "treasury" in col_lower or "fund" in col_lower or "reserve" in col_lower:
                treasury_col = col
                break

        # Look for developer columns
        for col in self.developer_columns:
            col_lower = col.lower()
            if "developer" in col_lower or "engineer" in col_lower or "contributor" in col_lower or "core dev" in col_lower:
                developer_col = col
                break

        # Get values
        treasury_value = self._get_numeric_value(row.get(treasury_col))
        developers_value = self._get_numeric_value(row.get(developer_col))

        # Store values in explanation
        explanation["treasury_column"] = treasury_col
        explanation["developers_column"] = developer_col
        explanation["treasury_value"] = treasury_value
        explanation["developers_value"] = developers_value

        # Calculate score
        if treasury_value is not None and developers_value is not None:
            # Normalize values (assuming reasonable ranges)
            normalized_treasury = min(100, treasury_value / 1_000_000)  # Up to $1M treasury
            normalized_developers = min(100, developers_value * 10)  # Up to 10 developers

            # Combine scores (weighted average)
            score = 0.6 * normalized_treasury + 0.4 * normalized_developers
            explanation["score"] = round(score, 2)
            explanation["method"] = "Calculated from treasury and developer metrics"
            return score, explanation

        # If only treasury data is available
        elif treasury_value is not None:
            score = min(100, treasury_value / 1_000_000)
            explanation["score"] = round(score, 2)
            explanation["method"] = "Calculated from treasury data"
            return score, explanation

        # If only developer data is available
        elif developers_value is not None:
            score = min(100, developers_value * 10)
            explanation["score"] = round(score, 2)
            explanation["method"] = "Calculated from developer metrics"
            return score, explanation

        # If no data is available
        else:
            default_score = 25.0  # Assuming low sustainability without data
            explanation["score"] = default_score
            explanation["method"] = "Insufficient sustainability data, using default"
            return default_score, explanation
    def calculate_transaction_activity(self, row: pd.Series) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate transaction activity score (0-100) based on transaction metrics.

        Args:
            row: A pandas Series representing a single row of the DataFrame.

        Returns:
            Tuple of (transactions_score, explanation_dict).
        """
        explanation = {
            "transactions_column": None,
            "transactions_value": None,
            "score": None,
            "method": "No valid data"
        }

        # Find transactions column
        transactions_col = None

        # Look for transaction columns
        for col in self.transaction_columns:
            col_lower = col.lower()
            if "transactions" in col_lower or "tx" in col_lower or "trade" in col_lower:
                transactions_col = col
                break

        # Get value
        transactions_value = self._get_numeric_value(row.get(transactions_col))

        # Store values in explanation
        explanation["transactions_column"] = transactions_col
        explanation["transactions_value"] = transactions_value

        # Calculate score
        if transactions_value is not None:
            # Normalize value (assuming reasonable ranges)
            normalized_transactions = min(100, transactions_value / 100_000)  # Up to 100k transactions

            # Assign score
            score = normalized_transactions
            explanation["score"] = round(score, 2)
            explanation["method"] = "Calculated from transaction metrics"
            return score, explanation

        # If no data is available
        else:
            default_score = 10.0  # Assuming low transaction activity without data
            explanation["score"] = default_score
            explanation["method"] = "Insufficient transaction data, using default"
            return default_score, explanation
    
    def calculate_comprehensive_score(self, csv_path: str = "eqs_results.csv") -> Tuple[float, Dict[str, Any]]:
        """
        Calculate a comprehensive score (0-100) for each row in the DataFrame.

        Returns:
            Tuple of (comprehensive_score, explanation_dict)
        """
        
        comprehensive_scores = []
        explanations = []
        self.results_explanations = []
        
        # Iterate through each row to calculate scores
        print(self.df.head())
        for index, row in self.df.iterrows():
            
            # Get the individual scores
            stability_score, stability_explanation = self.calculate_stability_score(row)
            revenue_diversification_score, revenue_diversification_explanation = self.calculate_revenue_diversification(row)
            user_efficiency_score, user_efficiency_explanation = self.calculate_user_efficiency(row)
            sustainability_score, sustainability_explanation = self.calculate_sustainability(row)
            transaction_activity_score, transaction_activity_explanation = self.calculate_transaction_activity(row)
            
            # Weighted average for the comprehensive score
            comprehensive_score = (
                0.3 * stability_score +
                0.25 * revenue_diversification_score +
                0.2 * user_efficiency_score +
                0.15 * sustainability_score +
                0.1 * transaction_activity_score
            )
            
            # Store results
            comprehensive_scores.append(comprehensive_score)
            
            # Generate detailed explanation
            explanation = {
                "stability": stability_explanation,
                "revenue_diversification": revenue_diversification_explanation,
                "user_efficiency": user_efficiency_explanation,
                "sustainability": sustainability_explanation,
                "transaction_activity": transaction_activity_explanation,
                "comprehensive_score": round(comprehensive_score, 2)
            }
            
            explanations.append(explanation)
            
        # Add comprehensive scores to the dataframe
        self.df["comprehensive_score"] = comprehensive_scores
        
        # Return comprehensive scores and explanations
        self.results_explanations = explanations
        #export the results to the output path
        export_results_to_csv(self.results_explanations, self.df, csv_path)
        return comprehensive_scores, self.results_explanations
