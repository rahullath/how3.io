import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import argparse

class FairValueAnalyzer:
    """
    A specialized class for analyzing fair value and valuation metrics
    for cryptocurrency projects.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe containing crypto project metrics."""
        self.df = df
        self._clean_data()
        self.valuation_columns = self._find_valuation_columns()
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
    
    def _find_valuation_columns(self) -> Dict[str, List[str]]:
        """Find columns related to valuation metrics."""
        valuation_columns = {
            'market_cap': [],
            'revenue': [],
            'multiples': [],
            'token_supply': [],
            'token_distribution': []
        }
        
        # Map column name patterns to metrics
        column_patterns = {
            'market_cap': [
                'Market cap', 'Marketcap', 'Mcap', 'Fully diluted'
            ],
            'revenue': [
                'Revenue', 'Earnings', 'Fees', 'Profit', 'Supply-side fees'
            ],
            'multiples': [
                'P/S ratio', 'P/F ratio', 'Price to sales', 'Price to fees',
                'Market cap to revenue'
            ],
            'token_supply': [
                'Circulating supply', 'Maximum supply', 'Total supply',
                'Supply ratio', 'Token inflation'
            ],
            'token_distribution': [
                'Tokenholders', 'Concentration', 'Treasury', 'Net treasury',
                'Distribution', 'Gini'
            ]
        }
        
        # Search for columns matching patterns
        for metric, patterns in column_patterns.items():
            for col in self.df.columns:
                col_str = str(col).lower()
                if any(pattern.lower() in col_str for pattern in patterns):
                    valuation_columns[metric].append(col)
        
        # Log found columns
        for metric, cols in valuation_columns.items():
            print(f"Found {len(cols)} columns for {metric}: {cols[:3]}")
            
        return valuation_columns
    
    def _get_market_sector_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Define which metrics to prioritize for each market sector.
        Returns a dictionary of sector -> metric -> weight mappings.
        """
        sector_metrics = {
            'Lending': {
                'market_cap': 0.20,
                'revenue': 0.30,
                'multiples': 0.35,
                'token_supply': 0.10,
                'token_distribution': 0.05
            },
            'Exchanges (DEX)': {
                'market_cap': 0.20,
                'revenue': 0.30,
                'multiples': 0.35,
                'token_supply': 0.10,
                'token_distribution': 0.05
            },
            'Derivative exchanges': {
                'market_cap': 0.20,
                'revenue': 0.30,
                'multiples': 0.35,
                'token_supply': 0.10,
                'token_distribution': 0.05
            },
            'Blockchains (L1)': {
                'market_cap': 0.25,
                'revenue': 0.25,
                'multiples': 0.30,
                'token_supply': 0.10,
                'token_distribution': 0.10
            },
            'Blockchains (L2)': {
                'market_cap': 0.25,
                'revenue': 0.25,
                'multiples': 0.30,
                'token_supply': 0.10,
                'token_distribution': 0.10
            },
            'Stablecoin issuers': {
                'market_cap': 0.20,
                'revenue': 0.30,
                'multiples': 0.35,
                'token_supply': 0.05,
                'token_distribution': 0.10
            },
            # Default weights for any other sector
            'default': {
                'market_cap': 0.20,
                'revenue': 0.25,
                'multiples': 0.35,
                'token_supply': 0.10,
                'token_distribution': 0.10
            }
        }
        
        return sector_metrics
        
    def _get_best_column(self, metric: str, prefer_latest: bool = True) -> Optional[str]:
        """
        Get the most appropriate column for a specific metric.
        
        Args:
            metric: The metric category ('market_cap', 'revenue', etc.)
            prefer_latest: Whether to prefer columns with 'latest' or 'current' data
            
        Returns:
            The column name or None if no suitable column found
        """
        if metric not in self.valuation_columns or not self.valuation_columns[metric]:
            return None
            
        columns = self.valuation_columns[metric]
        
        # Preferred time periods in order
        periods = ['Latest', '24h', '7d', '30d', '90d', '180d', '365d']
        
        if prefer_latest:
            # Try to find columns matching preferred periods
            for period in periods:
                for col in columns:
                    col_str = str(col).lower()
                    if period.lower() in col_str:
                        return col
                    
            # For multiples, prefer circulating over fully diluted
            if metric == 'multiples':
                for col in columns:
                    col_str = str(col).lower()
                    if 'circulating' in col_str:
                        return col
                        
            # Special case for revenue - prefer annualized if available
            if metric == 'revenue':
                for col in columns:
                    col_str = str(col).lower()
                    if 'annualized' in col_str:
                        return col
        
        # If no preferred period found, return the first column
        return columns[0] if columns else None
    
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
    
    def calculate_token_supply_ratio(self, row_idx: int) -> Tuple[Optional[float], Dict]:
        """
        Calculate the token supply ratio (circulating/maximum) and inflation potential.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Tuple of (ratio, explanation_dict)
        """
        # Get circulating supply
        circ_supply = None
        max_supply = None
        
        # Try to find columns with circulating and maximum supply
        for col in self.valuation_columns['token_supply']:
            col_str = str(col).lower()
            if 'circulating' in col_str:
                circ_supply = self._get_numeric_value(self._get_value(row_idx, col))
            elif 'maximum' in col_str or 'total' in col_str:
                max_supply = self._get_numeric_value(self._get_value(row_idx, col))
        
        # Prepare explanation dictionary
        explanation = {
            'circulating_supply': circ_supply,
            'maximum_supply': max_supply,
            'ratio': None,
            'inflation_potential': None,
            'ratio_percentile': None,
            'score_impact': None
        }
        
        # Calculate ratio if both values are available
        if circ_supply is not None and max_supply is not None and max_supply > 0:
            ratio = circ_supply / max_supply
            explanation['ratio'] = ratio
            
            # Calculate inflation potential (higher is worse)
            inflation_potential = 1 - ratio
            explanation['inflation_potential'] = inflation_potential
            
            # Get sector comparisons
            sector = self.df.iloc[row_idx]['Market sector']
            sector_mask = self.df['Market sector'] == sector
            sector_projects = self.df[sector_mask]
            
            # Calculate ratios for all projects in the sector
            all_ratios = []
            for idx, proj_row in sector_projects.iterrows():
                if idx != row_idx:  # Skip the current project
                    proj_circ = None
                    proj_max = None
                    for col in self.valuation_columns['token_supply']:
                        col_str = str(col).lower()
                        if 'circulating' in col_str:
                            proj_circ = self._get_numeric_value(proj_row.get(col))
                        elif 'maximum' in col_str or 'total' in col_str:
                            proj_max = self._get_numeric_value(proj_row.get(col))
                    
                    if proj_circ is not None and proj_max is not None and proj_max > 0:
                        all_ratios.append(proj_circ / proj_max)
            
            if all_ratios:
                # Calculate percentile (higher ratio is better - less inflation risk)
                percentile = sum(1 for r in all_ratios if r <= ratio) / len(all_ratios) * 100
                explanation['ratio_percentile'] = percentile
                
                # Calculate score impact (higher percentile = higher score)
                if percentile >= 75:
                    score_impact = 10  # Very good ratio - high percentile
                elif percentile >= 50:
                    score_impact = 5   # Above average ratio
                elif percentile >= 25:
                    score_impact = 0   # Below average but not terrible
                else:
                    score_impact = -10  # Very low ratio - lots of inflation risk
                
                explanation['score_impact'] = score_impact
                return score_impact, explanation
        
        return None, explanation
    
    def calculate_token_distribution(self, row_idx: int) -> Tuple[Optional[float], Dict]:
        """
        Calculate token distribution score based on tokenholders or concentration data.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Tuple of (score, explanation_dict)
        """
        # Get tokenholders or other distribution metric
        token_holders = None
        distribution_col = None
        
        # Try to find columns with token distribution info
        for col in self.valuation_columns['token_distribution']:
            col_str = str(col).lower()
            val = self._get_numeric_value(self._get_value(row_idx, col))
            if val is not None:
                token_holders = val
                distribution_col = col
                break
        
        # Prepare explanation dictionary
        explanation = {
            'metric_used': distribution_col,
            'value': token_holders,
            'percentile': None,
            'score_impact': None
        }
        
        if token_holders is None or distribution_col is None:
            return None, explanation
        
        # Get sector comparisons
        sector = self.df.iloc[row_idx]['Market sector']
        sector_mask = self.df['Market sector'] == sector
        sector_projects = self.df[sector_mask]
        
        # Get values for all projects in the sector
        all_values = []
        for idx, proj_row in sector_projects.iterrows():
            if idx != row_idx:  # Skip the current project
                val = self._get_numeric_value(proj_row.get(distribution_col))
                if val is not None:
                    all_values.append(val)
        
        if not all_values:
            return None, explanation
        
        # Calculate percentile (higher is better for token holders)
        percentile = sum(1 for v in all_values if v <= token_holders) / len(all_values) * 100
        explanation['percentile'] = percentile
        
        # More holders is better - less concentration
        if 'holder' in distribution_col.lower():
            # High number of holders is good (less concentration)
            if percentile >= 75:
                score_impact = 5  # Very high number of holders
            elif percentile >= 50:
                score_impact = 2  # Above average number
            elif percentile >= 25:
                score_impact = 0  # Below average but not terrible
            else:
                score_impact = -5  # Very low number of holders
        else:
            # For other metrics (like Gini coefficient), might need to reverse
            # This is a placeholder - would need to customize based on actual metrics
            score_impact = 0
        
        explanation['score_impact'] = score_impact
        return score_impact, explanation
    
    def calculate_ps_ratio_score(self, row_idx: int) -> Tuple[Optional[float], Dict]:
        """
        Calculate a score based on the price-to-sales (P/S) ratio.
        Lower P/S ratio generally indicates undervaluation.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Tuple of (score, explanation_dict)
        """
        # Get P/S ratio
        ps_ratio = None
        ps_col = None
        
        # Define market_cap_col and revenue_col right away to avoid reference errors
        market_cap_col = self._get_best_column('market_cap')
        revenue_col = self._get_best_column('revenue')
        
        # Try to find best column for P/S ratio
        for col in self.valuation_columns['multiples']:
            col_str = str(col).lower()
            if 'p/s' in col_str or 'price to sales' in col_str or 'ps ratio' in col_str:
                val = self._get_numeric_value(self._get_value(row_idx, col))
                if val is not None:
                    ps_ratio = val
                    ps_col = col
                    break
        
        # If P/S ratio not found directly, try to calculate it
        if ps_ratio is None and market_cap_col and revenue_col:
            market_cap = self._get_numeric_value(self._get_value(row_idx, market_cap_col))
            revenue = self._get_numeric_value(self._get_value(row_idx, revenue_col))
            
            if market_cap is not None and revenue is not None and revenue > 0:
                ps_ratio = market_cap / revenue
                ps_col = f"Calculated from {market_cap_col} / {revenue_col}"
        
        # Prepare explanation dictionary
        explanation = {
            'metric_used': ps_col,
            'ps_ratio': ps_ratio,
            'percentile': None,
            'score': None,
            'interpretation': None
        }
        
        if ps_ratio is None:
            return None, explanation
        
        # Get sector comparisons
        sector = self.df.iloc[row_idx]['Market sector']
        sector_mask = self.df['Market sector'] == sector
        sector_projects = self.df[sector_mask]
        
        # Get P/S ratios for all projects in the sector
        all_ratios = []
        
        # Try using the same column first
        if ps_col:
            for idx, proj_row in sector_projects.iterrows():
                if idx != row_idx:  # Skip the current project
                    if isinstance(ps_col, str) and 'Calculated' in ps_col:
                        # Need to calculate for other projects too
                        m_cap = self._get_numeric_value(proj_row.get(market_cap_col))
                        rev = self._get_numeric_value(proj_row.get(revenue_col))
                        if m_cap is not None and rev is not None and rev > 0:
                            all_ratios.append(m_cap / rev)
                    else:
                        val = self._get_numeric_value(proj_row.get(ps_col))
                        if val is not None:
                            all_ratios.append(val)
        
        if not all_ratios:
            # If we couldn't get ratios using the same column, try a more general approach
            for idx, proj_row in sector_projects.iterrows():
                if idx != row_idx:  # Skip the current project
                    # Try all multiple columns
                    for col in self.valuation_columns['multiples']:
                        col_str = str(col).lower()
                        if 'p/s' in col_str or 'price to sales' in col_str or 'ps ratio' in col_str:
                            val = self._get_numeric_value(proj_row.get(col))
                            if val is not None:
                                all_ratios.append(val)
                                break
                    
                    # If still no ratio, try to calculate it
                    if not all_ratios and market_cap_col and revenue_col:
                        m_cap = self._get_numeric_value(proj_row.get(market_cap_col))
                        rev = self._get_numeric_value(proj_row.get(revenue_col))
                        if m_cap is not None and rev is not None and rev > 0:
                            all_ratios.append(m_cap / rev)
        
        if not all_ratios:
            return None, explanation
        
        # For P/S ratio, lower is better, so we need to invert the percentile calculation
        percentile = sum(1 for r in all_ratios if r >= ps_ratio) / len(all_ratios) * 100
        explanation['percentile'] = percentile
        
        # Calculate sector statistics
        sector_avg = np.mean(all_ratios)
        sector_median = np.median(all_ratios)
        
        explanation['sector_stats'] = {
            'average': sector_avg,
            'median': sector_median,
            'min': min(all_ratios),
            'max': max(all_ratios),
            'count': len(all_ratios)
        }
        
        # Convert to score (0-100)
        # For P/S ratio, lower is better, so high percentile (our calculation) = good score
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
        
        # Add interpretation
        if ps_ratio < sector_median * 0.5:
            explanation['interpretation'] = "Significantly undervalued compared to sector median"
        elif ps_ratio < sector_median * 0.8:
            explanation['interpretation'] = "Moderately undervalued compared to sector median"
        elif ps_ratio < sector_median * 1.2:
            explanation['interpretation'] = "Fairly valued compared to sector median"
        elif ps_ratio < sector_median * 2:
            explanation['interpretation'] = "Moderately overvalued compared to sector median"
        else:
            explanation['interpretation'] = "Significantly overvalued compared to sector median"
        
        return score, explanation
    
    def calculate_revenue_ratio_score(self, row_idx: int) -> Tuple[Optional[float], Dict]:
        """
        Calculate a score based on revenue in relation to market cap.
        Higher revenue-to-market-cap ratio indicates better value.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Tuple of (score, explanation_dict)
        """
        # Get revenue and market cap
        revenue = None
        market_cap = None
        
        revenue_col = self._get_best_column('revenue')
        market_cap_col = self._get_best_column('market_cap')
        
        if revenue_col and market_cap_col:
            revenue = self._get_numeric_value(self._get_value(row_idx, revenue_col))
            market_cap = self._get_numeric_value(self._get_value(row_idx, market_cap_col))
        
        # Prepare explanation dictionary
        explanation = {
            'revenue_metric': revenue_col,
            'market_cap_metric': market_cap_col,
            'revenue': revenue,
            'market_cap': market_cap,
            'revenue_to_market_cap': None,
            'percentile': None,
            'score': None
        }
        
        if revenue is None or market_cap is None or market_cap <= 0:
            return None, explanation
        
        # Calculate revenue-to-market-cap ratio
        rev_to_mcap = revenue / market_cap
        explanation['revenue_to_market_cap'] = rev_to_mcap
        
        # Get sector comparisons
        sector = self.df.iloc[row_idx]['Market sector']
        sector_mask = self.df['Market sector'] == sector
        sector_projects = self.df[sector_mask]
        
        # Calculate ratio for all projects in the sector
        all_ratios = []
        for idx, proj_row in sector_projects.iterrows():
            if idx != row_idx:  # Skip the current project
                proj_rev = self._get_numeric_value(proj_row.get(revenue_col))
                proj_mcap = self._get_numeric_value(proj_row.get(market_cap_col))
                
                if proj_rev is not None and proj_mcap is not None and proj_mcap > 0:
                    all_ratios.append(proj_rev / proj_mcap)
        
        if not all_ratios:
            return None, explanation
        
        # Calculate percentile (higher ratio is better)
        percentile = sum(1 for r in all_ratios if r <= rev_to_mcap) / len(all_ratios) * 100
        explanation['percentile'] = percentile
        
        # Convert to score (0-100)
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
        
        # Add sector statistics
        explanation['sector_stats'] = {
            'average': np.mean(all_ratios),
            'median': np.median(all_ratios),
            'min': min(all_ratios),
            'max': max(all_ratios),
            'count': len(all_ratios)
        }
        
        return score, explanation
    
    def calculate_fair_value_score(self, row_idx: int) -> Dict:
        """
        Calculate the overall fair value score for a project.
        
        Args:
            row_idx: Index of the project in the dataframe
            
        Returns:
            Dictionary with fair value score and explanations
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
            'valuation_category': None,
            'weights_used': weights,
            'explanation': "Fair value score based on weighted average of valuation metrics"
        }
        
        # Calculate P/S ratio score
        ps_score, ps_explanation = self.calculate_ps_ratio_score(row_idx)
        if ps_score is not None:
            results['metrics']['ps_ratio'] = {
                'score': ps_score,
                'weight': weights['multiples'],
                'explanation': ps_explanation
            }
        
        # Calculate revenue ratio score
        rev_score, rev_explanation = self.calculate_revenue_ratio_score(row_idx)
        if rev_score is not None:
            results['metrics']['revenue_ratio'] = {
                'score': rev_score,
                'weight': weights['revenue'],
                'explanation': rev_explanation
            }
        
        # Calculate token supply ratio score (inflation potential)
        token_score, token_explanation = self.calculate_token_supply_ratio(row_idx)
        if token_score is not None:
            # Convert token score from adjustment to actual score
            token_base_score = 50  # Neutral base
            adjusted_score = token_base_score + token_score
            adjusted_score = max(0, min(100, adjusted_score))  # Ensure in 0-100 range
            
            results['metrics']['token_supply'] = {
                'score': adjusted_score,
                'weight': weights['token_supply'],
                'explanation': token_explanation
            }
        
        # Calculate token distribution score
        dist_score, dist_explanation = self.calculate_token_distribution(row_idx)
        if dist_score is not None:
            # Convert distribution score from adjustment to actual score
            dist_base_score = 50  # Neutral base
            adjusted_score = dist_base_score + dist_score
            adjusted_score = max(0, min(100, adjusted_score))  # Ensure in 0-100 range
            
            results['metrics']['token_distribution'] = {
                'score': adjusted_score,
                'weight': weights['token_distribution'],
                'explanation': dist_explanation
            }
        
        # Calculate overall score
        weighted_scores = []
        for metric, data in results['metrics'].items():
            weighted_scores.append((data['score'], data['weight']))
        
        if weighted_scores:
            overall_score = sum(score * weight for score, weight in weighted_scores) / sum(weight for _, weight in weighted_scores)
            results['overall_score'] = overall_score
            
            # Determine valuation category
            if overall_score >= 70:
                category = "Undervalued"
            elif overall_score >= 40:
                category = "Fairly Valued"
            else:
                category = "Overvalued"
                
            results['valuation_category'] = category
        
        return results
    
    def analyze_all_projects(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze fair value for all projects in the dataset.
        
        Args:
            output_file: Optional path to save results CSV
            
        Returns:
            DataFrame with fair value scores
        """
        results = []
        
        # Process each project
        for idx, row in self.df.iterrows():
            project_name = row.get('Project')
            
            # Skip if no project name
            if pd.isna(project_name):
                continue
                
            print(f"Analyzing fair value for {project_name}...")
            
            # Calculate fair value score
            project_results = self.calculate_fair_value_score(idx)
            
            # Add to results
            results.append({
                'Project': project_name,
                'Market Sector': project_results['sector'],
                'Valuation Category': project_results.get('valuation_category'),
                'Details': project_results
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            # Save only the main columns, not the details
            columns_to_save = ['Project', 'Market Sector', 'Fair Value Score']
            
            # Only include Valuation Category if it exists in the DataFrame
            if 'Valuation Category' in results_df.columns:
                columns_to_save.append('Valuation Category')
                
            # Filter to only columns that exist
            save_cols = [col for col in columns_to_save if col in results_df.columns]
            
            # Save the results
            save_df = results_df[save_cols]
            save_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df
    
    def visualize_sector_valuations(self, output_file: Optional[str] = None) -> None:
        """
        Create visualizations of fair value by sector.
        
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
        results_df = results_df.dropna(subset=['Fair Value Score'])
        
        # Plot 1: Box plot of scores by sector
        plt.subplot(2, 1, 1)
        sector_data = results_df.groupby('Market Sector')['Fair Value Score'].agg(['mean', 'count'])
        sector_data = sector_data.sort_values('mean', ascending=False)
        
        # Only show sectors with at least 3 projects
        sectors_to_plot = sector_data[sector_data['count'] >= 3].index.tolist()
        
        if sectors_to_plot:
            plot_df = results_df[results_df['Market Sector'].isin(sectors_to_plot)]
            sns.boxplot(x='Market Sector', y='Fair Value Score', data=plot_df, 
                        order=sectors_to_plot)
            plt.xticks(rotation=45, ha='right')
            plt.title('Fair Value Scores by Market Sector')
            plt.tight_layout()
        
        # Plot 2: Top 20 most undervalued projects
        plt.subplot(2, 1, 2)
        top_projects = results_df.nlargest(20, 'Fair Value Score')
        
        sns.barplot(x='Fair Value Score', y='Project', data=top_projects, 
                    hue='Market Sector', dodge=False)
        plt.title('Top 20 Most Undervalued Projects')
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
            
        plt.show()
    
    def create_valuation_table(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Create a table of valuation metrics for all projects.
        
        Args:
            output_file: Optional path to save results CSV
            
        Returns:
            DataFrame with detailed valuation metrics
        """
        # Get results for all projects
        results_df = self.analyze_all_projects()
        
        if results_df.empty:
            print("No results to create valuation table")
            return pd.DataFrame()
        
        # Extract key metrics from the Details field
        detailed_results = []
        
        for _, row in results_df.iterrows():
            project = row['Project']
            sector = row['Market Sector']
            fair_value_score = row['Fair Value Score']
            valuation_category = row.get('Valuation Category')
            details = row.get('Details', {})
            
            result = {
                'Project': project,
                'Market Sector': sector,
                'Fair Value Score': fair_value_score,
                'Valuation Category': valuation_category
            }
            
            # Extract P/S ratio
            ps_details = details.get('metrics', {}).get('ps_ratio', {}).get('explanation', {})
            if ps_details:
                result['P/S Ratio'] = ps_details.get('ps_ratio')
                result['P/S Percentile'] = ps_details.get('percentile')
                result['P/S Interpretation'] = ps_details.get('interpretation')
                
                # Add sector comparison
                sector_stats = ps_details.get('sector_stats', {})
                if sector_stats:
                    result['Sector Avg P/S'] = sector_stats.get('average')
                    result['Sector Median P/S'] = sector_stats.get('median')
            
            # Extract revenue ratio
            rev_details = details.get('metrics', {}).get('revenue_ratio', {}).get('explanation', {})
            if rev_details:
                result['Revenue/Market Cap'] = rev_details.get('revenue_to_market_cap')
                result['Revenue'] = rev_details.get('revenue')
                result['Market Cap'] = rev_details.get('market_cap')
            
            # Extract token supply metrics
            token_details = details.get('metrics', {}).get('token_supply', {}).get('explanation', {})
            if token_details:
                result['Circulating/Max Supply'] = token_details.get('ratio')
                result['Inflation Potential'] = token_details.get('inflation_potential')
            
            detailed_results.append(result)
        
        # Create DataFrame
        detailed_df = pd.DataFrame(detailed_results)
        
        # Save to file if specified
        if output_file:
            detailed_df.to_csv(output_file, index=False)
            print(f"Detailed valuation table saved to {output_file}")
        
        return detailed_df


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
    """Main function to run the fair value analysis."""
    parser = argparse.ArgumentParser(description='Analyze fair value for crypto projects')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to data file (CSV or Excel)')
    parser.add_argument('--output', type=str, default='fair_value_results.csv',
                        help='Path to output CSV file')
    parser.add_argument('--detailed', type=str, 
                        help='Path to save detailed metrics table')
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
        analyzer = FairValueAnalyzer(df)
        
        # Run analysis
        results = analyzer.analyze_all_projects(args.output)
        
        # Create detailed metrics table if requested
        if args.detailed:
            analyzer.create_valuation_table(args.detailed)
        
        # Create visualizations if requested
        if args.visualize:
            analyzer.visualize_sector_valuations(args.viz_output)
        
        print(f"Analysis complete. Found {len(results)} projects with fair value data.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()