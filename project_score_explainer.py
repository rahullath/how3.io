import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from typing import Dict, List, Optional, Any, Tuple

class ProjectScoreExplainer:
    """Tool to explain score calculations for a specific crypto project."""
    
    def __init__(self, data_file: str, results_file: Optional[str] = None):
        """Initialize with raw data and optional results file."""
        self.data_file = data_file
        self.results_file = results_file
        self.df = None
        self.results_df = None
        self.metric_columns = None
        self.sector_metrics = None
        
        # Load data
        self.load_data()
        
        # Initialize metric mappings
        self.initialize_metrics()
        
        # Load results if available
        if results_file and os.path.exists(results_file):
            try:
                self.results_df = pd.read_csv(results_file)
                print(f"Loaded results data with {self.results_df.shape[0]} projects")
            except Exception as e:
                print(f"Error loading results file: {e}")
    
    def load_data(self) -> None:
        """Load the raw crypto project data."""
        try:
            # Determine file type from extension
            if self.data_file.lower().endswith('.csv'):
                self.df = pd.read_csv(self.data_file, dtype=str)
            elif self.data_file.lower().endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.data_file, dtype=str)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            print(f"Loaded raw data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            
            # Clean data
            self._clean_data()
        except Exception as e:
            print(f"Error loading data file: {e}")
    
    def _clean_data(self) -> None:
        """Clean and prepare the data for analysis."""
        if self.df is None:
            return
        
        # Replace infinity values with NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Handle unnamed columns
        if 'Unnamed: 1' in self.df.columns and 'Unnamed: 2' in self.df.columns:
            self.df.rename(columns={
                'Unnamed: 1': 'Project', 
                'Unnamed: 2': 'Market sector'
            }, inplace=True)
        
        # Ensure we have the critical columns
        required_cols = ['Project', 'Market sector']
        for col in required_cols:
            if col not in self.df.columns:
                # Try to find a similarly named column
                for existing_col in self.df.columns:
                    if col.lower() in str(existing_col).lower():
                        self.df.rename(columns={existing_col: col}, inplace=True)
                        print(f"Renamed column '{existing_col}' to '{col}'")
                        break
                
                # Check header row if column still missing
                if col not in self.df.columns and self.df.shape[0] > 0:
                    first_row = self.df.iloc[0]
                    for idx, value in enumerate(first_row):
                        if col.lower() in str(value).lower():
                            col_name = self.df.columns[idx]
                            self.df.rename(columns={col_name: col}, inplace=True)
                            print(f"Found '{col}' in first row, renamed column {col_name}")
                            break
        
        # Convert numeric columns where possible
        for col in self.df.columns:
            if col not in ['Project', 'Market sector', 'Listing Date']:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass
    
    def initialize_metrics(self) -> None:
        """Initialize metric mappings and find columns."""
        self.metric_columns = self._find_metric_columns(self._map_metrics())
        self.sector_metrics = self._get_market_sector_metrics()
    
    def _map_metrics(self) -> Dict:
        """Define metrics needed for each benchmark."""
        # User Growth Metrics
        user_growth_metrics = {
            'daily_active_addresses': ['Active addresses (daily)', 'Active users (daily)', 
                                      'Daily active users', 'DAU', 'Active'],
            'transaction_volume': ['Transaction volume', 'Transfer volume', 'Volume', 'tx volume'],
            'transaction_growth': ['Transaction volume 365d change', 'YoY Growth', 'Annual Growth', '365d change'],
            'bridge_volume': ['Bridge deposits', 'Bridge volume', 'Deposits', 'Bridge inflow']
        }
        
        # Earnings Quality Metrics
        earnings_quality_metrics = {
            'revenue': ['Revenue', 'Fees', 'Earnings', 'Protocol revenue'],
            'revenue_stability': ['Revenue 30d change', '30d trend', 'Revenue stability', 'Volatility'],
            'protocol_fees': ['Protocol fees', 'Fees', 'Trading fees'],
            'transaction_fees': ['Average transaction fee', 'Fee', 'Transaction fee', 'Fee per transaction']
        }
        
        # Fair Value Metrics
        fair_value_metrics = {
            'market_cap': ['Market cap (circulating)', 'Market cap', 'Mcap'],
            'market_cap_to_revenue': ['P/S ratio', 'PS ratio', 'Price to sales', 'Market cap to revenue'],
            'token_inflation': ['Maximum supply', 'Circulating supply', 'Supply ratio', 'Inflation'],
            'token_concentration': ['Tokenholders', 'Holders', 'Concentration', 'Gini']
        }
        
        # Safety & Stability Metrics
        safety_stability_metrics = {
            'validator_count': ['Number of validators', 'Validators', 'Nodes', 'Node count'],
            'governance_participation': ['Voting incentives', 'Governance', 'Voting', 'Participation'],
            'developer_activity': ['Core developers', 'Code commits', 'Developers', 'GitHub activity'],
            'security_incidents': ['Security', 'Incidents', 'Exploits', 'Hacks']
        }
        
        # Combined metrics mapping
        all_metrics = {
            'User Growth': user_growth_metrics,
            'Earnings Quality': earnings_quality_metrics,
            'Fair Value': fair_value_metrics,
            'Safety & Stability': safety_stability_metrics
        }
        
        return all_metrics
    
    def _get_market_sector_metrics(self) -> Dict:
        """Define which metrics to use for each market sector."""
        sector_metrics = {
            'Lending': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['protocol_fees', 'revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            },
            'DeFi': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['protocol_fees', 'revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            },
            'CeFi': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['transaction_fees', 'revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['governance_participation']
            },
            'GameFi': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['protocol_fees', 'revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['governance_participation']
            },
            'DePIN': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            },
            'L1/L2': {
                'User Growth': ['transaction_volume', 'bridge_volume'],
                'Earnings Quality': ['transaction_fees', 'protocol_fees'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            },
            'Infrastructure': {
                'User Growth': ['transaction_volume', 'bridge_volume'],
                'Earnings Quality': ['protocol_fees', 'transaction_fees'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            },
            'AI Agents': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['revenue'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['governance_participation', 'developer_activity']
            },
            # Default for any other sectors
            'default': {
                'User Growth': ['daily_active_addresses', 'transaction_volume'],
                'Earnings Quality': ['revenue', 'protocol_fees'],
                'Fair Value': ['market_cap_to_revenue'],
                'Safety & Stability': ['validator_count', 'governance_participation']
            }
        }
        
        # Add aliases for similar sector names
        sector_aliases = {
            'Blockchains (L1)': 'L1/L2',
            'Blockchains (L2)': 'L1/L2',
            'Layer 1': 'L1/L2',
            'Layer 2': 'L1/L2',
            'Decentralized Finance': 'DeFi',
            'Centralized Finance': 'CeFi',
            'Stablecoin issuers': 'CeFi',
            'Gaming': 'GameFi',
            'AI': 'AI Agents',
            'Artificial Intelligence': 'AI Agents'
        }
        
        # Add aliases to the main dictionary
        for alias, target in sector_aliases.items():
            if target in sector_metrics:
                sector_metrics[alias] = sector_metrics[target]
        
        return sector_metrics
    
    def _find_metric_columns(self, main_metrics: Dict) -> Dict:
        """Find columns in the dataset that match our metrics."""
        if self.df is None:
            return {}
        
        metric_columns = {}
        
        # Check for header row in data if most columns are unnamed
        header_row_idx = None
        col_to_metric = {}
        
        uses_unnamed_cols = sum(1 for col in self.df.columns if 'Unnamed:' in str(col)) > 5
        if uses_unnamed_cols:
            print("Detecting header row in unnamed columns...")
            # Try to find a header row with metric keywords
            for idx, row in self.df.iterrows():
                metric_matches = sum(1 for val in row if isinstance(val, str) and 
                                    any(m in str(val).lower() for m in 
                                        ['active', 'market cap', 'transaction', 'revenue']))
                if metric_matches >= 3:
                    header_row_idx = idx
                    print(f"Found potential header row at index {idx}")
                    
                    # Build column mapping
                    for col_idx, col_name in enumerate(self.df.columns):
                        if col_idx < len(row):
                            metric_value = row[col_idx]
                            if isinstance(metric_value, str) and len(metric_value) > 3:
                                col_to_metric[col_name] = metric_value
                    break
        
        # Create metric mappings
        for category, metrics in main_metrics.items():
            metric_columns[category] = {}
            
            for metric_key, metric_names in metrics.items():
                if metric_names is None:
                    continue
                
                found_cols = []
                
                # Process each metric name
                for name in metric_names:
                    # Try exact match first
                    exact_matches = [col for col in self.df.columns 
                                    if name == col or (isinstance(col, str) and col.strip() == name)]
                    if exact_matches:
                        found_cols.extend(exact_matches)
                        continue
                    
                    # Try partial match on column names
                    partial_matches = [col for col in self.df.columns 
                                      if isinstance(col, str) and name.lower() in col.lower()]
                    if partial_matches:
                        found_cols.extend(partial_matches)
                        continue
                    
                    # For unnamed columns, check the values in header row
                    if uses_unnamed_cols and header_row_idx is not None:
                        for col, value in col_to_metric.items():
                            if isinstance(value, str) and name.lower() in value.lower():
                                found_cols.append(col)
                                break
                
                # Store unique columns found for this metric
                if found_cols:
                    found_cols = list(set(found_cols))  # Remove duplicates
                    metric_columns[category][metric_key] = found_cols
        
        # Debug info
        for category, metrics in metric_columns.items():
            print(f"Found {len(metrics)} metrics for {category}")
            for metric_key, cols in metrics.items():
                print(f"  - {metric_key}: {cols[:2]}{'...' if len(cols) > 2 else ''}")
        
        return metric_columns
    
    def get_project_index(self, project_name: str) -> Optional[int]:
        """Find the row index for a project in the raw data."""
        if self.df is None:
            return None
        
        # Check for exact match first
        exact_matches = self.df[self.df['Project'] == project_name]
        if not exact_matches.empty:
            return exact_matches.index[0]
        
        # Try case-insensitive match
        for idx, row in self.df.iterrows():
            if 'Project' in row and isinstance(row['Project'], str) and row['Project'].lower() == project_name.lower():
                return idx
        
        # Try partial match if needed
        for idx, row in self.df.iterrows():
            if 'Project' in row and isinstance(row['Project'], str) and project_name.lower() in row['Project'].lower():
                print(f"Found partial match: '{row['Project']}' for '{project_name}'")
                return idx
        
        return None
    
    def get_best_column(self, category: str, metric_key: str, prefer_latest: bool = True) -> Optional[str]:
        """Get the best column for a specific metric."""
        if self.metric_columns is None or category not in self.metric_columns or metric_key not in self.metric_columns[category]:
            return None
        
        columns = self.metric_columns[category][metric_key]
        
        if not columns:
            return None
        
        # If we want the latest/current value
        if prefer_latest:
            # Priority order for current values
            for preferred in ['Latest', 'Current', '24h avg', '7d avg']:
                for col in columns:
                    if preferred.lower() in str(col).lower():
                        return col
        
        # Fall back to the first column
        return columns[0]
    
    def get_value(self, row_idx: int, col_name: str, default: Any = None) -> Any:
        """Safely extract a value from the dataframe."""
        if self.df is None or col_name not in self.df.columns:
            return default
        
        value = self.df.iloc[row_idx][col_name]
        
        if pd.isna(value):
            return default
        
        # Try to convert to numeric if possible
        if isinstance(value, str):
            value = value.strip()
            # Handle various numeric formats
            try:
                if any(char in value.upper() for char in ['E', 'e']) or value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return float(value)
            except (ValueError, TypeError):
                pass
        
        return value
    
    def get_numeric_value(self, value: Any) -> Optional[float]:
        """Convert a value to numeric format safely."""
        if value is None:
            return None
            
        if isinstance(value, (int, float)) and not np.isnan(value):
            return float(value)
            
        if isinstance(value, str):
            # Clean string and convert to float
            try:
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            except (ValueError, TypeError):
                return None
                
        return None
    
    def calculate_percentile_score(self, value: Any, all_values: List, reverse: bool = False) -> Tuple[Optional[float], Dict]:
        """
        Calculate a percentile-based score with detailed explanation.
        
        Returns:
            Tuple of (score, explanation_dict)
        """
        explanation = {
            'value': value,
            'num_comparison_values': len(all_values) if all_values else 0,
            'percentile': None,
            'formula': 'Score based on percentile ranking against peer projects',
            'reverse_metric': reverse,
            'raw_numeric': None,
            'better_than_count': None,
            'score_range': None
        }
        
        if value is None or pd.isna(value) or not all_values:
            return None, explanation
        
        # Convert value to numeric
        numeric_value = self.get_numeric_value(value)
        explanation['raw_numeric'] = numeric_value
        
        if numeric_value is None:
            return None, explanation
        
        # Process all reference values
        numeric_values = []
        for v in all_values:
            num_v = self.get_numeric_value(v)
            if num_v is not None:
                numeric_values.append(num_v)
        
        explanation['num_valid_comparisons'] = len(numeric_values)
        
        if not numeric_values:
            return None, explanation
        
        # Find percentile
        numeric_values = sorted(numeric_values)
        
        if reverse:
            # For reverse metrics, lower is better
            rank = sum(1 for v in numeric_values if v >= numeric_value)
            explanation['comparison_direction'] = 'Lower is better'
        else:
            # For normal metrics, higher is better
            rank = sum(1 for v in numeric_values if v <= numeric_value)
            explanation['comparison_direction'] = 'Higher is better'
        
        explanation['better_than_count'] = rank
        explanation['total_count'] = len(numeric_values)
        
        percentile = (rank / len(numeric_values)) * 100
        explanation['percentile'] = percentile
        
        # Convert to score on our scale
        if percentile >= 90:
            score_range = '90-100 (Excellent)'
            score = 90 + (percentile - 90) * (10/10)
        elif percentile >= 70:
            score_range = '70-89 (Strong)'
            score = 70 + (percentile - 70) * (20/20)
        elif percentile >= 30:
            score_range = '40-69 (Average)'
            score = 40 + (percentile - 30) * (30/40)
        elif percentile >= 10:
            score_range = '20-39 (Weak)'
            score = 20 + (percentile - 10) * (20/20)
        else:
            score_range = '0-19 (Poor)'
            score = percentile * (20/10)
        
        explanation['score_range'] = score_range
        explanation['score_calculation'] = f"Raw percentile: {percentile:.2f}% → Score: {score:.2f}"
        
        return score, explanation
    
    def explain_project_score(self, project_name: str, verbose: bool = True) -> Dict:
        """
        Generate a detailed explanation of score calculations for a project.
        
        Args:
            project_name: Name of the project to analyze
            verbose: Whether to print detailed analysis during calculation
            
        Returns:
            Dictionary containing detailed score explanations
        """
        if self.df is None:
            print("Error: No data loaded")
            return {}
        
        # Find the project in the data
        row_idx = self.get_project_index(project_name)
        
        if row_idx is None:
            print(f"Error: Project '{project_name}' not found in the dataset")
            return {}
        
        # Get project info
        actual_name = self.df.iloc[row_idx]['Project']
        market_sector = self.df.iloc[row_idx]['Market sector']
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"SCORE CALCULATION FOR: {actual_name}")
            print(f"Market Sector: {market_sector}")
            print(f"{'='*50}")
        
        # Get metrics relevant to this sector
        if market_sector in self.sector_metrics:
            metrics_to_use = self.sector_metrics[market_sector]
        else:
            metrics_to_use = self.sector_metrics['default']
            
        if verbose:
            print(f"\nMetrics used for {market_sector} sector:")
            for category, metrics in metrics_to_use.items():
                print(f"  {category}: {', '.join(metrics)}")
        
        # Initialize results
        results = {
            'project': actual_name,
            'market_sector': market_sector,
            'categories': {},
            'overall_score': None,
            'valuation': None
        }
        
        # Process each category
        for category, metrics in metrics_to_use.items():
            if verbose:
                print(f"\n{'-'*50}")
                print(f"{category} SCORE CALCULATION:")
                print(f"{'-'*50}")
            
            category_scores = []
            category_explanations = {}
            
            for metric_key in metrics:
                # Get the appropriate column for this metric
                col_name = self.get_best_column(category, metric_key)
                
                if col_name:
                    # Get value for this project
                    value = self.get_value(row_idx, col_name)
                    
                    if value is not None and not pd.isna(value):
                        # Get all values for this metric in this sector
                        sector_mask = self.df['Market sector'] == market_sector
                        all_values = self.df.loc[sector_mask, col_name].dropna().tolist()
                        
                        # Determine if lower is better
                        reverse = metric_key in ['market_cap_to_revenue', 'token_inflation']
                        
                        # Calculate score with explanation
                        score, explanation = self.calculate_percentile_score(value, all_values, reverse)
                        
                        if score is not None:
                            category_scores.append(score)
                            explanation['metric_name'] = metric_key
                            explanation['column_used'] = col_name
                            explanation['data_value'] = value
                            category_explanations[metric_key] = explanation
                            
                            if verbose:
                                print(f"\nMetric: {metric_key}")
                                print(f"  Column used: {col_name}")
                                print(f"  Value: {value}")
                                print(f"  Compared against: {len(all_values)} {market_sector} projects")
                                if reverse:
                                    print(f"  Direction: Lower is better")
                                else:
                                    print(f"  Direction: Higher is better")
                                print(f"  Percentile: {explanation['percentile']:.2f}%") 
                                print(f"  Score range: {explanation['score_range']}")
                                print(f"  Final score: {score:.2f}")
                else:
                    if verbose:
                        print(f"\nMetric: {metric_key} - No suitable column found")
            
            # Calculate category score
            if category_scores:
                category_score = sum(category_scores) / len(category_scores)
                if verbose:
                    print(f"\n{category} Category Score: {category_score:.2f} (average of {len(category_scores)} metrics)")
            else:
                category_score = None
                if verbose:
                    print(f"\n{category} Category Score: Not available (no valid metrics found)")
            
            # Store category results
            results['categories'][category] = {
                'score': category_score,
                'metrics': category_explanations,
                'num_metrics_used': len(category_scores)
            }
        
        # Calculate overall score
        category_scores = [cat_data['score'] for cat_data in results['categories'].values() 
                         if cat_data['score'] is not None]
        
        if category_scores:
            overall_score = sum(category_scores) / len(category_scores)
            results['overall_score'] = overall_score
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"OVERALL SCORE: {overall_score:.2f}")
                print(f"Based on {len(category_scores)} categories")
                print(f"{'='*50}")
        else:
            if verbose:
                print("\nCould not calculate overall score - insufficient data")
        
        # Determine valuation classification
        if self.results_df is not None and 'Overall Score' in self.results_df.columns:
            top_threshold = self.results_df['Overall Score'].quantile(0.75)
            bottom_threshold = self.results_df['Overall Score'].quantile(0.25)
            
            if overall_score is not None:
                if overall_score >= top_threshold:
                    valuation = 'Undervalued'
                elif overall_score <= bottom_threshold:
                    valuation = 'Overvalued'
                else:
                    valuation = 'Aptly Valued'
                
                results['valuation'] = valuation
                results['valuation_thresholds'] = {
                    'undervalued_threshold': top_threshold,
                    'overvalued_threshold': bottom_threshold
                }
                
                if verbose:
                    print(f"\nValuation Classification: {valuation}")
                    print(f"Thresholds: Undervalued ≥ {top_threshold:.2f}, Overvalued ≤ {bottom_threshold:.2f}")
        
        return results
    
    def visualize_project_score(self, project_results: Dict, output_path: Optional[str] = None) -> None:
        """
        Create visualizations to explain a project's score breakdown.
        
        Args:
            project_results: Output from explain_project_score()
            output_path: Optional path to save the visualization
        """
        if not project_results:
            print("No results to visualize")
            return
        
        project_name = project_results['project']
        
        # Set up the visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        
        # Create a figure with multiple plots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Category Scores Bar Chart
        ax1 = plt.subplot(2, 2, 1)
        categories = []
        scores = []
        
        for cat, data in project_results['categories'].items():
            if data['score'] is not None:
                categories.append(cat)
                scores.append(data['score'])
        
        if scores:
            colors = ['#2ecc71' if s >= 70 else '#3498db' if s >= 40 else '#e74c3c' for s in scores]
            bars = ax1.bar(categories, scores, color=colors)
            ax1.set_ylim(0, 100)
            ax1.set_title(f'Category Scores for {project_name}')
            ax1.set_ylabel('Score (0-100)')
            
            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{score:.1f}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No category scores available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Individual Metrics Breakdown
        ax2 = plt.subplot(2, 2, 2)
        metrics = []
        metric_scores = []
        metric_categories = []
        
        for cat, data in project_results['categories'].items():
            for metric, metric_data in data['metrics'].items():
                if 'score_calculation' in metric_data:
                    metrics.append(metric)
                    score_parts = metric_data['score_calculation'].split('→')
                    if len(score_parts) > 1:
                        try:
                            score = float(score_parts[1].strip().split(':')[1])
                            metric_scores.append(score)
                            metric_categories.append(cat)
                        except:
                            continue
        
        if metric_scores:
            category_colors = {
                'User Growth': '#3498db',
                'Earnings Quality': '#2ecc71',
                'Fair Value': '#9b59b6',
                'Safety & Stability': '#f39c12'
            }
            
            colors = [category_colors.get(cat, '#95a5a6') for cat in metric_categories]
            y_pos = range(len(metrics))
            
            # Sort by score 
            sorted_data = sorted(zip(metrics, metric_scores, colors), key=lambda x: x[1])
            metrics = [x[0] for x in sorted_data]
            metric_scores = [x[1] for x in sorted_data]
            colors = [x[2] for x in sorted_data]
            
            bars = ax2.barh(y_pos, metric_scores, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.set_xlim(0, 100)
            ax2.set_title(f'Individual Metric Scores')
            ax2.set_xlabel('Score (0-100)')
            
            # Add value labels
            for i, v in enumerate(metric_scores):
                ax2.text(v + 1, i, f'{v:.1f}', va='center')
            
            # Add a legend for categories
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, label=cat)
                for cat, color in category_colors.items()
                if cat in metric_categories
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
        else:
            ax1.text(0.5, 0.5, 'No category scores available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Individual Metrics Breakdown
        ax2 = plt.subplot(2, 2, 2)
        metrics = []
        metric_scores = []
        metric_categories = []
        
        for cat, data in project_results['categories'].items():
            for metric, metric_data in data['metrics'].items():
                if 'score_calculation' in metric_data:
                    metrics.append(metric)
                    score_parts = metric_data['score_calculation'].split('→')
                    if len(score_parts) > 1:
                        try:
                            score = float(score_parts[1].strip().split(':')[1])
                            metric_scores.append(score)
                            metric_categories.append(cat)
                        except:
                            continue
        
        if metric_scores:
            category_colors = {
                'User Growth': '#3498db',
                'Earnings Quality': '#2ecc71',
                'Fair Value': '#9b59b6',
                'Safety & Stability': '#f39c12'
            }
            
            colors = [category_colors.get(cat, '#95a5a6') for cat in metric_categories]
            y_pos = range(len(metrics))
            
            # Sort by score 
            sorted_data = sorted(zip(metrics, metric_scores, colors), key=lambda x: x[1])
            metrics = [x[0] for x in sorted_data]
            metric_scores = [x[1] for x in sorted_data]
            colors = [x[2] for x in sorted_data]
            
            bars = ax2.barh(y_pos, metric_scores, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.set_xlim(0, 100)
            ax2.set_title(f'Individual Metric Scores')
            ax2.set_xlabel('Score (0-100)')
            
            # Add value labels
            for i, v in enumerate(metric_scores):
                ax2.text(v + 1, i, f'{v:.1f}', va='center')
            
            # Add a legend for categories
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, label=cat)
                for cat, color in category_colors.items()
                if cat in metric_categories
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
        else:
            ax2.text(0.5, 0.5, 'No individual metrics available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Percentile Position Visualization
        ax3 = plt.subplot(2, 2, 3)
        
        # Find a representative metric with percentile data
        example_metric = None
        example_cat = None
        
        for cat, data in project_results['categories'].items():
            for metric, metric_data in data['metrics'].items():
                if 'percentile' in metric_data and metric_data['percentile'] is not None:
                    example_metric = metric
                    example_cat = cat
                    percentile = metric_data['percentile']
                    metric_name = metric_data['metric_name']
                    break
            if example_metric:
                break
        
        if example_metric:
            # Create a visual representation of where this project sits in the distribution
            x = np.linspace(0, 100, 1000)
            y = np.exp(-(x - 50)**2 / 500)  # Normal-ish distribution centered at 50
            
            ax3.plot(x, y, color='#bdc3c7')
            ax3.fill_between(x, 0, y, color='#ecf0f1', alpha=0.7)
            
            # Mark the percentile position
            height = np.exp(-(percentile - 50)**2 / 500)
            ax3.scatter([percentile], [height], color='#e74c3c', s=100, zorder=5)
            ax3.vlines(percentile, 0, height, colors='#e74c3c', linestyles='--')
            
            # Add labels and annotations
            ax3.text(percentile, height + 0.01, f'{percentile:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', color='#e74c3c')
            
            # Shade the distribution regions
            ax3.fill_between(x[x < 10], 0, y[x < 10], color='#e74c3c', alpha=0.3)
            ax3.fill_between(x[(x >= 10) & (x < 30)], 0, y[(x >= 10) & (x < 30)], color='#e67e22', alpha=0.3)
            ax3.fill_between(x[(x >= 30) & (x < 70)], 0, y[(x >= 30) & (x < 70)], color='#f1c40f', alpha=0.3)
            ax3.fill_between(x[(x >= 70) & (x < 90)], 0, y[(x >= 70) & (x < 90)], color='#2ecc71', alpha=0.3)
            ax3.fill_between(x[x >= 90], 0, y[x >= 90], color='#27ae60', alpha=0.3)
            
            ax3.set_xlim(0, 100)
            ax3.set_ylim(0, max(y) * 1.2)
            ax3.set_title(f'Percentile Position: {metric_name}')
            ax3.set_xlabel('Percentile (lower → higher)')
            ax3.get_yaxis().set_visible(False)
            
            # Add text labels for the regions
            ax3.text(5, max(y) * 1.1, 'Bottom 10%', ha='center', fontsize=8)
            ax3.text(20, max(y) * 1.1, '10-30%', ha='center', fontsize=8)
            ax3.text(50, max(y) * 1.1, '30-70%', ha='center', fontsize=8)
            ax3.text(80, max(y) * 1.1, '70-90%', ha='center', fontsize=8)
            ax3.text(95, max(y) * 1.1, 'Top 10%', ha='center', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No percentile data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Overall Score Gauge
        ax4 = plt.subplot(2, 2, 4, polar=True)
        
        if 'overall_score' in project_results and project_results['overall_score'] is not None:
            overall_score = project_results['overall_score']
            
            # Create a gauge-like visualization
            # Convert score (0-100) to radians (0-pi)
            theta = np.linspace(0, np.pi, 100)
            
            # Create the background
            ax4.bar(theta, 1.0, width=np.pi/50, bottom=0.65, 
                  color=plt.cm.viridis(theta/np.pi), alpha=0.7)
            
            # Mark the overall score
            score_theta = overall_score * np.pi / 100
            ax4.scatter([score_theta], [0.825], s=400, color='#e74c3c', zorder=5)
            
            # Add a needle/pointer
            ax4.plot([score_theta, score_theta], [0, 0.65], color='#c0392b', linewidth=4)
            
            # Customize the appearance
            ax4.set_title(f'Overall Score: {overall_score:.1f}/100', pad=15)
            ax4.set_yticks([])
            
            # Only show the top half of the polar plot
            ax4.set_thetamin(0)
            ax4.set_thetamax(180)
            
            # Add score range labels
            ax4.text(np.pi * 0.05, 0.4, 'Poor', ha='center', va='center', fontsize=9)
            ax4.text(np.pi * 0.25, 0.4, 'Weak', ha='center', va='center', fontsize=9)
            ax4.text(np.pi * 0.5, 0.4, 'Average', ha='center', va='center', fontsize=9)
            ax4.text(np.pi * 0.75, 0.4, 'Strong', ha='center', va='center', fontsize=9)
            ax4.text(np.pi * 0.95, 0.4, 'Excellent', ha='center', va='center', fontsize=9)
            
            # Add valuation classification if available
            if 'valuation' in project_results and project_results['valuation'] is not None:
                valuation = project_results['valuation']
                color_map = {
                    'Undervalued': '#2ecc71',  # Green
                    'Aptly Valued': '#f39c12',  # Orange
                    'Overvalued': '#e74c3c'    # Red
                }
                color = color_map.get(valuation, '#7f8c8d')
                
                ax4.text(np.pi/2, 0.9, valuation, 
                      ha='center', va='center', fontsize=12, fontweight='bold',
                      bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.5'))
        else:
            ax4.text(0, 0, 'Overall score not available', 
                   ha='center', va='center')
        
        # Add a title for the entire figure
        plt.suptitle(f'Score Analysis: {project_name}', fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_path:
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
        
        plt.show()
    
    def export_explanation_to_html(self, project_results: Dict, output_path: str) -> None:
        """
        Export the score explanation to an HTML file for interactive viewing.
        
        Args:
            project_results: Output from explain_project_score()
            output_path: Path to save the HTML file
        """
        if not project_results:
            print("No results to export")
            return
        
        project_name = project_results['project']
        sector = project_results['market_sector']
        overall_score = project_results.get('overall_score')
        valuation = project_results.get('valuation')
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Score Analysis: {project_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #34495e;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .score-container {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }}
                .score-circle {{
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
                    margin-right: 15px;
                }}
                .excellent {{
                    background-color: #27ae60;
                }}
                .strong {{
                    background-color: #2ecc71;
                }}
                .average {{
                    background-color: #f1c40f;
                }}
                .weak {{
                    background-color: #e67e22;
                }}
                .poor {{
                    background-color: #e74c3c;
                }}
                .score-details {{
                    flex: 1;
                }}
                .metric-row {{
                    display: flex;
                    border-bottom: 1px solid #eee;
                    padding: 10px 0;
                }}
                .metric-name {{
                    flex: 1;
                    font-weight: bold;
                }}
                .metric-value {{
                    flex: 1;
                    text-align: center;
                }}
                .metric-score {{
                    flex: 1;
                    text-align: right;
                }}
                .valuation {{
                    font-size: 18px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin-top: 10px;
                }}
                .undervalued {{
                    background-color: #d5f5e3;
                    color: #27ae60;
                }}
                .aptly-valued {{
                    background-color: #fef9e7;
                    color: #f39c12;
                }}
                .overvalued {{
                    background-color: #fadbd8;
                    color: #e74c3c;
                }}
                .score-bar-container {{
                    width: 100%;
                    background-color: #f1f1f1;
                    border-radius: 5px;
                    margin: 5px 0;
                }}
                .score-bar {{
                    height: 25px;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                }}
                .percentile-box {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Score Analysis: {project_name}</h1>
                <p>Market Sector: {sector}</p>
            </div>
        """
        
        # Overall Score Section
        if overall_score is not None:
            # Determine color class based on score
            if overall_score >= 90:
                color_class = "excellent"
            elif overall_score >= 70:
                color_class = "strong"
            elif overall_score >= 40:
                color_class = "average"
            elif overall_score >= 20:
                color_class = "weak"
            else:
                color_class = "poor"
                
            html_content += f"""
            <div class="card">
                <h2>Overall Score</h2>
                <div class="score-container">
                    <div class="score-circle {color_class}">{overall_score:.1f}</div>
                    <div class="score-details">
                        <p>Based on analysis across multiple categories</p>
                        <div class="score-bar-container">
                            <div class="score-bar {color_class}" style="width: {min(overall_score, 100)}%;">
                                {overall_score:.1f}/100
                            </div>
                        </div>
                    </div>
                </div>
            """
            
            # Add valuation classification if available
            if valuation:
                val_class = valuation.lower().replace(" ", "-")
                html_content += f"""
                <div class="valuation {val_class}">
                    Valuation Classification: {valuation}
                </div>
                """
                
                # Add threshold information if available
                if 'valuation_thresholds' in project_results:
                    thresholds = project_results['valuation_thresholds']
                    html_content += f"""
                    <p>Thresholds: 
                       Undervalued ≥ {thresholds['undervalued_threshold']:.2f}, 
                       Overvalued ≤ {thresholds['overvalued_threshold']:.2f}
                    </p>
                    """
            
            html_content += "</div>"  # Close the overall score card
        
        # Category Scores Section
        html_content += "<h2>Category Scores</h2>"
        
        for category, data in project_results['categories'].items():
            if data['score'] is not None:
                cat_score = data['score']
                
                # Determine color class
                if cat_score >= 90:
                    cat_color_class = "excellent"
                elif cat_score >= 70:
                    cat_color_class = "strong"
                elif cat_score >= 40:
                    cat_color_class = "average"
                elif cat_score >= 20:
                    cat_color_class = "weak"
                else:
                    cat_color_class = "poor"
                
                html_content += f"""
                <div class="card">
                    <h3>{category}</h3>
                    <div class="score-container">
                        <div class="score-circle {cat_color_class}">{cat_score:.1f}</div>
                        <div class="score-details">
                            <p>Based on {data['num_metrics_used']} metrics</p>
                            <div class="score-bar-container">
                                <div class="score-bar {cat_color_class}" style="width: {min(cat_score, 100)}%;">
                                    {cat_score:.1f}/100
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h4>Metrics Used:</h4>
                """
                
                # Add individual metrics
                for metric, metric_data in data['metrics'].items():
                    metric_score = None
                    # Extract score from score_calculation
                    if 'score_calculation' in metric_data:
                        score_parts = metric_data['score_calculation'].split('→')
                        if len(score_parts) > 1:
                            try:
                                metric_score = float(score_parts[1].strip().split(':')[1])
                            except:
                                pass
                    
                    if metric_score is not None:
                        # Determine color class for metric
                        if metric_score >= 90:
                            metric_color_class = "excellent"
                        elif metric_score >= 70:
                            metric_color_class = "strong"
                        elif metric_score >= 40:
                            metric_color_class = "average"
                        elif metric_score >= 20:
                            metric_color_class = "weak"
                        else:
                            metric_color_class = "poor"
                        
                        html_content += f"""
                        <div class="metric-row">
                            <div class="metric-name">{metric}</div>
                            <div class="metric-value">{metric_data['data_value']}</div>
                            <div class="metric-score">
                                <div class="score-bar-container" style="width: 100px; display: inline-block;">
                                    <div class="score-bar {metric_color_class}" style="width: {min(metric_score, 100)}%; font-size: 0.8em;">
                                        {metric_score:.1f}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """
                        
                        # Add percentile information
                        if 'percentile' in metric_data and metric_data['percentile'] is not None:
                            html_content += f"""
                            <div class="percentile-box">
                                <p><strong>Percentile:</strong> {metric_data['percentile']:.2f}%</p>
                                <p><strong>Comparison:</strong> Better than {metric_data['better_than_count']} out of {metric_data['total_count']} {sector} projects</p>
                                <p><strong>Direction:</strong> {metric_data['comparison_direction']}</p>
                                <p><strong>Score Range:</strong> {metric_data['score_range']}</p>
                            </div>
                            """
                
                html_content += "</div>"  # Close the category card
        
        # Close the HTML content
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML explanation saved to {output_path}")
        except Exception as e:
            print(f"Error saving HTML explanation: {e}")


def main():
    """Main function to run the explainer tool."""
    parser = argparse.ArgumentParser(description='Explain score calculations for a specific crypto project.')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to raw data file (CSV or Excel)')
    parser.add_argument('--results', type=str, 
                        help='Path to results CSV from analysis')
    parser.add_argument('--project', type=str, required=True, 
                        help='Name of the project to explain')
    parser.add_argument('--output', type=str, 
                        help='Path to save visualization (PNG)')
    parser.add_argument('--html', type=str, 
                        help='Path to save HTML explanation')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print detailed calculation steps')
    
    args = parser.parse_args()
    
    # Initialize explainer
    explainer = ProjectScoreExplainer(args.data, args.results)
    
    # Generate explanation
    results = explainer.explain_project_score(args.project, args.verbose)
    
    # Visualize results if requested
    if args.output and results:
        explainer.visualize_project_score(results, args.output)
    
    # Export HTML explanation if requested
    if args.html and results:
        explainer.export_explanation_to_html(results, args.html)


if __name__ == "__main__":
    # Example usage with hardcoded paths for convenience
    import sys
    
    # If no arguments are provided, use defaults
    if len(sys.argv) == 1:
        # Try to find input files
        data_paths = [
            "crypto_data.xlsx",
            "crypto_data.csv",
            "ttmasterdata_20250327.xlsx",
            "tt-master-data_2025-03-27.xlsx",
            "tt-master-data_2025-03-27.csv"
        ]
        
        results_paths = [
            "crypto_analysis_results.csv",
            "crypto_valuation_results.csv",
            "results.csv"
        ]
        
        # Find first existing data file
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        # Find first existing results file
        results_file = None
        for path in results_paths:
            if os.path.exists(path):
                results_file = path
                break
        
        if data_file:
            print(f"Using data file: {data_file}")
            sys.argv.extend(["--data", data_file])
            
            if results_file:
                print(f"Using results file: {results_file}")
                sys.argv.extend(["--results", results_file])
            
            # Ask for project name
            project = input("Enter project name to analyze: ")
            sys.argv.extend(["--project", project])
            
            # Default output paths
            sys.argv.extend(["--output", f"{project}_score_analysis.png"])
            sys.argv.extend(["--html", f"{project}_score_explanation.html"])
            sys.argv.extend(["--verbose"])
        else:
            print("No data file detected. Please specify with --data.")
            sys.exit(1)
    
    main()
