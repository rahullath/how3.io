import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import os
import sys

class UserGrowthScoreExplainer:
    """A specialized class for explaining user growth scores for cryptocurrency projects."""
    
    def __init__(self, df: pd.DataFrame, results_df: Optional[pd.DataFrame] = None):
        """Initialize with the raw data DataFrame and optional results DataFrame."""
        if 'Project' not in df.columns and 'project' not in df.columns:
            print("Error: The dataset does not contain a 'Project' column.")
            print("Available columns:", df.columns.tolist())
            return
        self.df = df
        self.results_df = results_df
        self._standardize_columns()
        self._clean_data()
        self.growth_columns = self._find_growth_columns()
        self.sector_metrics = self._get_market_sector_metrics()
        
    def _standardize_columns(self) -> None:
        """Standardize column names to ensure compatibility."""
        if self.df is not None:
            self.df.columns = [col.strip().lower() for col in self.df.columns]
    
    def _clean_data(self) -> None:
        """Clean and prepare the data for analysis."""
        if self.df is None:
            return

        # Standardize column names
        self._standardize_columns()

        # Replace infinity values with NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
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
    
    def _get_value(self, row: pd.Series, col_name: str, default_val: Any = None) -> Any:
        """Safely get a value from a DataFrame row."""
        if col_name not in row.index:
            return default_val
            
        value = row[col_name]
        
        if pd.isna(value):
            return default_val
            
        return value

    def get_project_by_name(self, project_name: str) -> Optional[pd.Series]:
        """Find a project by name in the DataFrame."""
        if self.df is None:
            return None

        # Use the standardized column name
        if 'project' not in self.df.columns:
            return None

        matches = self.df[self.df['project'] == project_name]
        if not matches.empty:
            return matches.iloc[0]

        # Try case-insensitive match
        for _, row in self.df.iterrows():
            if 'project' in row and isinstance(row['project'], str) and row['project'].lower() == project_name.lower():
                return row

        return None
    
    def get_user_growth_explanation(self, project_name: str) -> Dict:
        """
        Generate a detailed explanation of user growth score for a specific project.
        
        Args:
            project_name: Name of the project to analyze
            
        Returns:
            Dictionary with detailed explanation
        """
        project_row = self.get_project_by_name(project_name)
        if project_row is None:
            return {'error': f"Project '{project_name}' not found in the dataset"}
            
        sector = project_row.get('Market sector', 'Unknown')
        
        # Get weights for this sector
        weights = self.sector_metrics.get(sector, self.sector_metrics['default'])
        
        # Initialize results
        explanation = {
            'project': project_name,
            'sector': sector,
            'weights': weights,
            'metrics': {},
            'overall_score': None,
            'growth_category': None,
            'explanation': ""
        }
        
        # Look up the overall score in the results DataFrame if available
        if self.results_df is not None:
            result_row = self.results_df[self.results_df['Project'] == project_name]
            if not result_row.empty:
                explanation['overall_score'] = result_row['User Growth Score'].iloc[0]
                explanation['growth_category'] = result_row['Growth Category'].iloc[0]
        
        # Analyze each metric
        for metric, weight in weights.items():
            # Get best column for this metric
            col_name = self._get_best_column(metric)
            if col_name is None:
                continue
                
            # Get value for this project
            value = self._get_value(project_row, col_name)
            numeric_value = self._get_numeric_value(value)
            
            if numeric_value is None:
                continue
                
            # Get sector comparison
            sector_mask = self.df['Market sector'] == sector
            sector_data = self.df[sector_mask]
            
            sector_values = []
            sector_projects = []
            for _, row in sector_data.iterrows():
                if row['Project'] != project_name:  # Skip the current project
                    proj_value = self._get_value(row, col_name)
                    proj_numeric = self._get_numeric_value(proj_value)
                    if proj_numeric is not None:
                        sector_values.append(proj_numeric)
                        sector_projects.append(row['Project'])
            
            # Calculate percentile
            if sector_values:
                percentile = sum(1 for v in sector_values if v <= numeric_value) / len(sector_values) * 100
                
                # Get projects with higher and lower values
                higher_projects = [proj for i, proj in enumerate(sector_projects) if sector_values[i] > numeric_value]
                lower_projects = [proj for i, proj in enumerate(sector_projects) if sector_values[i] < numeric_value]
                
                # Sort by closeness to the project's value
                higher_projects = sorted(higher_projects, 
                                         key=lambda p: abs(self._get_numeric_value(self._get_value(sector_data[sector_data['Project'] == p].iloc[0], col_name)) - numeric_value))
                lower_projects = sorted(lower_projects,
                                        key=lambda p: abs(self._get_numeric_value(self._get_value(sector_data[sector_data['Project'] == p].iloc[0], col_name)) - numeric_value))
                
                # Take the closest few projects
                higher_examples = higher_projects[:3]
                lower_examples = lower_projects[:3]
                
                # Add to explanation
                explanation['metrics'][metric] = {
                    'column': col_name,
                    'value': numeric_value,
                    'percentile': percentile,
                    'weight': weight,
                    'sector_average': np.mean(sector_values) if sector_values else None,
                    'sector_median': np.median(sector_values) if sector_values else None,
                    'sector_comparison': {
                        'higher_than': lower_examples,
                        'lower_than': higher_examples
                    }
                }
                
                # Get some descriptive text
                if percentile > 90:
                    comparison_text = f"exceptional compared to other {sector} projects"
                elif percentile > 75:
                    comparison_text = f"very strong compared to other {sector} projects"
                elif percentile > 50:
                    comparison_text = f"above average compared to other {sector} projects"
                elif percentile > 25:
                    comparison_text = f"below average compared to other {sector} projects"
                else:
                    comparison_text = f"relatively low compared to other {sector} projects"
                
                explanation['metrics'][metric]['description'] = f"{metric.replace('_', ' ').title()} is {comparison_text}"
        
        # Generate overall explanation
        if explanation['overall_score'] is not None:
            score = explanation['overall_score']
            category = explanation['growth_category']
            
            if score >= 80:
                explanation['explanation'] = (
                    f"{project_name} shows exceptional user growth metrics with a score of {score:.1f}/100. "
                    f"It outperforms most peers in its {sector} category, particularly in "
                )
            elif score >= 65:
                explanation['explanation'] = (
                    f"{project_name} demonstrates strong user growth with a score of {score:.1f}/100. "
                    f"It performs well among {sector} projects, with notable strength in "
                )
            elif score >= 45:
                explanation['explanation'] = (
                    f"{project_name} shows steady user growth metrics with a score of {score:.1f}/100. "
                    f"Its performance is competitive within the {sector} sector, particularly in "
                )
            elif score >= 25:
                explanation['explanation'] = (
                    f"{project_name} exhibits moderate user growth with a score of {score:.1f}/100. "
                    f"While not leading its {sector} category, it shows some positive indicators in "
                )
            else:
                explanation['explanation'] = (
                    f"{project_name} has limited user growth metrics with a score of {score:.1f}/100. "
                    f"It faces challenges compared to other {sector} projects, though it does show potential in "
                )
            
            # Add top metrics to explanation
            top_metrics = sorted(
                [(m, data) for m, data in explanation['metrics'].items() if 'percentile' in data],
                key=lambda x: x[1]['percentile'],
                reverse=True
            )
            
            if top_metrics:
                top_metric_names = [m.replace('_', ' ').title() for m, _ in top_metrics[:2]]
                if len(top_metric_names) > 1:
                    explanation['explanation'] += f"{top_metric_names[0]} and {top_metric_names[1]}."
                else:
                    explanation['explanation'] += f"{top_metric_names[0]}."
            else:
                explanation['explanation'] += "certain areas."
        
        return explanation
        
    def visualize_user_growth_explanation(self, project_name: str, output_file: Optional[str] = None) -> None:
        """
        Create a visualization explaining the user growth score for a project.
        
        Args:
            project_name: Name of the project to visualize
            output_file: Optional file path to save the visualization
        """
        explanation = self.get_user_growth_explanation(project_name)
        if 'error' in explanation:
            print(explanation['error'])
            return
        
        # Setup figure
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"User Growth Analysis: {project_name}", fontsize=16, fontweight='bold')
        
        # Define colors
        colors = {
            'excellent': '#2ecc71',  # Green
            'strong': '#27ae60',    # Dark Green
            'average': '#f1c40f',   # Yellow
            'weak': '#e67e22',      # Orange
            'poor': '#e74c3c'       # Red
        }
        
        # 1. Overall Score gauge (top left)
        plt.subplot(2, 2, 1)
        overall_score = explanation.get('overall_score')
        if overall_score is not None:
            # Create a gauge chart
            category = explanation.get('growth_category', '')
            if 'Exceptional' in category:
                color = colors['excellent']
            elif 'Strong' in category:
                color = colors['strong']
            elif 'Steady' in category:
                color = colors['average']
            elif 'Slow' in category:
                color = colors['weak']
            else:
                color = colors['poor']
                
            plt.pie([overall_score, 100-overall_score], 
                    colors=[color, '#ecf0f1'], 
                    startangle=90, 
                    counterclock=False,
                    wedgeprops={'width': 0.3, 'edgecolor': 'w'})
            
            plt.text(0, 0, f"{overall_score:.1f}", ha='center', va='center', fontsize=24, fontweight='bold')
            plt.text(0, -0.2, f"{category}", ha='center', va='center', fontsize=12)
            
            plt.axis('equal')
            plt.title('User Growth Score', fontsize=14)
        else:
            plt.text(0.5, 0.5, "No overall score available", ha='center', va='center')
            plt.axis('off')
        
        # 2. Metric breakdown (top right)
        plt.subplot(2, 2, 2)
        metrics = explanation.get('metrics', {})
        if metrics:
            metric_names = []
            metric_scores = []
            metric_colors = []
            
            for metric, data in metrics.items():
                if 'percentile' in data:
                    percentile = data['percentile']
                    metric_names.append(metric.replace('_', ' ').title())
                    
                    # Convert percentile to score (same formula as in the main class)
                    if percentile >= 90:
                        score = 90 + (percentile - 90) * (10/10)  # 90-100
                        color = colors['excellent']
                    elif percentile >= 70:
                        score = 70 + (percentile - 70) * (20/20)  # 70-89
                        color = colors['strong']
                    elif percentile >= 30:
                        score = 40 + (percentile - 30) * (30/40)  # 40-69
                        color = colors['average']
                    elif percentile >= 10:
                        score = 20 + (percentile - 10) * (20/20)  # 20-39
                        color = colors['weak']
                    else:
                        score = percentile * (20/10)  # 0-19
                        color = colors['poor']
                    
                    metric_scores.append(score)
                    metric_colors.append(color)
            
            # Sort by score
            sorted_data = sorted(zip(metric_names, metric_scores, metric_colors), key=lambda x: x[1], reverse=True)
            metric_names = [x[0] for x in sorted_data]
            metric_scores = [x[1] for x in sorted_data]
            metric_colors = [x[2] for x in sorted_data]
            
            # Create horizontal bar chart
            y_pos = range(len(metric_names))
            plt.barh(y_pos, metric_scores, color=metric_colors)
            plt.yticks(y_pos, metric_names)
            plt.xlim(0, 100)
            plt.xlabel('Score')
            plt.title('Metric Scores', fontsize=14)
            
            # Add score labels
            for i, v in enumerate(metric_scores):
                plt.text(v + 1, i, f"{v:.1f}", va='center')
        else:
            plt.text(0.5, 0.5, "No metric data available", ha='center', va='center')
            plt.axis('off')
        
        # 3. Sector comparison (bottom)
        plt.subplot(2, 1, 2)
        if metrics:
            # Find the most relevant metric (highest weight)
            weights = explanation.get('weights', {})
            top_metric = max(metrics.keys() & weights.keys(), key=lambda m: weights[m], default=None)
            
            if top_metric and 'percentile' in metrics[top_metric]:
                data = metrics[top_metric]
                percentile = data['percentile']
                sector = explanation['sector']
                
                # Create a distribution-like visualization
                x = np.linspace(0, 100, 1000)
                y = np.exp(-(x - 50)**2 / 500)  # Normal-ish distribution centered at 50
                
                plt.plot(x, y, color='#bdc3c7')
                plt.fill_between(x, 0, y, color='#ecf0f1', alpha=0.7)
                
                # Mark the percentile position
                height = np.exp(-(percentile - 50)**2 / 500)
                plt.scatter([percentile], [height], color='#e74c3c', s=100, zorder=5)
                plt.vlines(percentile, 0, height, colors='#e74c3c', linestyles='--')
                
                # Add labels and annotations
                plt.text(percentile, height + 0.01, f'{percentile:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', color='#e74c3c')
                
                # Shade the distribution regions
                plt.fill_between(x[x < 10], 0, y[x < 10], color='#e74c3c', alpha=0.3)
                plt.fill_between(x[(x >= 10) & (x < 30)], 0, y[(x >= 10) & (x < 30)], color='#e67e22', alpha=0.3)
                plt.fill_between(x[(x >= 30) & (x < 70)], 0, y[(x >= 30) & (x < 70)], color='#f1c40f', alpha=0.3)
                plt.fill_between(x[(x >= 70) & (x < 90)], 0, y[(x >= 70) & (x < 90)], color='#2ecc71', alpha=0.3)
                plt.fill_between(x[x >= 90], 0, y[x >= 90], color='#27ae60', alpha=0.3)
                
                plt.xlim(0, 100)
                plt.ylim(0, max(y) * 1.2)
                plt.title(f'{top_metric.replace("_", " ").title()} Percentile in {sector} Sector', fontsize=14)
                plt.xlabel('Percentile (lower → higher)')
                plt.gca().get_yaxis().set_visible(False)
                
                # Add text labels for the regions
                plt.text(5, max(y) * 1.1, 'Poor', ha='center', fontsize=8)
                plt.text(20, max(y) * 1.1, 'Weak', ha='center', fontsize=8)
                plt.text(50, max(y) * 1.1, 'Average', ha='center', fontsize=8)
                plt.text(80, max(y) * 1.1, 'Strong', ha='center', fontsize=8)
                plt.text(95, max(y) * 1.1, 'Excellent', ha='center', fontsize=8)
                
                # Add comparison info
                higher_than = data['sector_comparison']['higher_than']
                lower_than = data['sector_comparison']['lower_than']
                
                comparison_text = f"Higher than: {', '.join(higher_than[:3])}" if higher_than else ""
                if lower_than:
                    comparison_text += f"\nLower than: {', '.join(lower_than[:3])}"
                
                plt.figtext(0.5, 0.05, comparison_text, ha='center', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            else:
                plt.text(0.5, 0.5, "No detailed sector comparison available", ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No sector comparison data available", ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"User growth visualization saved to {output_file}")
        
        plt.show()
    
    def generate_html_explanation(self, project_name: str, output_file: Optional[str] = None) -> str:
        """
        Generate an HTML report explaining user growth for a project.
        
        Args:
            project_name: Name of the project to explain
            output_file: Optional file path to save the HTML
            
        Returns:
            HTML string with the explanation
        """
        explanation = self.get_user_growth_explanation(project_name)
        if 'error' in explanation:
            html = f"<html><body><h1>Error</h1><p>{explanation['error']}</p></body></html>"
            return html
        
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>User Growth Analysis: {project_name}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .summary {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-left: 5px solid #3498db;
                    margin-bottom: 30px;
                }}
                .score-container {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .score-circle {{
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 24px;
                    margin-right: 20px;
                }}
                .excellent {{ background-color: #2ecc71; }}
                .strong {{ background-color: #27ae60; }}
                .steady {{ background-color: #f1c40f; }}
                .slow {{ background-color: #e67e22; }}
                .stagnant {{ background-color: #e74c3c; }}
                .score-details {{
                    flex-grow: 1;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .progress-container {{
                    width: 100%;
                    background-color: #f1f1f1;
                    border-radius: 4px;
                    margin: 5px 0;
                }}
                .progress-bar {{
                    height: 24px;
                    border-radius: 4px;
                    text-align: center;
                    line-height: 24px;
                    color: white;
                    font-weight: bold;
                }}
                .comparison {{
                    background-color: #eef8ff;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>User Growth Analysis: {project_name}</h1>
                <p>Market Sector: {explanation['sector']}</p>
            </div>
        """
        
        # Add summary section
        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p>{explanation.get('explanation', 'No summary available.')}</p>
            </div>
        """
        
        # Add overall score section
        overall_score = explanation.get('overall_score')
        category = explanation.get('growth_category', '')
        if overall_score is not None:
            # Determine CSS class based on category
            css_class = ''
            if 'Exceptional' in category:
                css_class = 'excellent'
            elif 'Strong' in category:
                css_class = 'strong'
            elif 'Steady' in category:
                css_class = 'steady'
            elif 'Slow' in category:
                css_class = 'slow'
            else:
                css_class = 'stagnant'
                
            html += f"""
                <h2>Overall User Growth Score</h2>
                <div class="score-container">
                    <div class="score-circle {css_class}">{overall_score:.1f}</div>
                    <div class="score-details">
                        <h3>{category}</h3>
                        <div class="progress-container">
                            <div class="progress-bar {css_class}" style="width: {min(overall_score, 100)}%;">
                                {overall_score:.1f}/100
                            </div>
                        </div>
                    </div>
                </div>
            """
            
        # Add metrics breakdown
        metrics = explanation.get('metrics', {})
        if metrics:
            html += f"""
                <h2>Growth Metrics Breakdown</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Percentile</th>
                        <th>Weight</th>
                        <th>Score</th>
                    </tr>
            """
            
            for metric, data in metrics.items():
                if 'percentile' in data:
                    percentile = data['percentile']
                    
                    # Convert percentile to score (same formula as in the main class)
                    if percentile >= 90:
                        score = 90 + (percentile - 90) * (10/10)  # 90-100
                        css_class = 'excellent'
                    elif percentile >= 70:
                        score = 70 + (percentile - 70) * (20/20)  # 70-89
                        css_class = 'strong'
                    elif percentile >= 30:
                        score = 40 + (percentile - 30) * (30/40)  # 40-69
                        css_class = 'steady'
                    elif percentile >= 10:
                        score = 20 + (percentile - 10) * (20/20)  # 20-39
                        css_class = 'slow'
                    else:
                        score = percentile * (20/10)  # 0-19
                        css_class = 'stagnant'
                    
                    html += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{data['value']:,}</td>
                            <td>{percentile:.1f}%</td>
                            <td>{data['weight'] * 100:.0f}%</td>
                            <td>
                                <div class="progress-container">
                                    <div class="progress-bar {css_class}" style="width: {min(score, 100)}%;">
                                        {score:.1f}
                                    </div>
                                </div>
                            </td>
                        </tr>
                    """
            
            html += """
                </table>
            """
            
            # Add sector comparison for top metric
            weights = explanation.get('weights', {})
            top_metric = max(metrics.keys() & weights.keys(), key=lambda m: weights[m], default=None)
            
            if top_metric and 'sector_comparison' in metrics[top_metric]:
                data = metrics[top_metric]
                html += f"""
                    <h2>Sector Comparison: {top_metric.replace('_', ' ').title()}</h2>
                    <div class="comparison">
                        <p>{data.get('description', '')}</p>
                        <p>
                            <strong>Value:</strong> {data['value']:,}<br>
                            <strong>Sector Average:</strong> {data.get('sector_average', 0):,.1f}<br>
                            <strong>Sector Median:</strong> {data.get('sector_median', 0):,.1f}<br>
                            <strong>Percentile:</strong> {data.get('percentile', 0):.1f}%
                        </p>
                """
                
                # Add comparison with other projects
                higher_than = data['sector_comparison'].get('higher_than', [])
                lower_than = data['sector_comparison'].get('lower_than', [])
                
                if higher_than:
                    html += f"""
                        <p><strong>Higher than:</strong> {', '.join(higher_than)}</p>
                    """
                
                if lower_than:
                    html += f"""
                        <p><strong>Lower than:</strong> {', '.join(lower_than)}</p>
                    """
                
                html += """
                    </div>
                """
        
        # Close the HTML
        html += """
        </body>
        </html>
        """
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML explanation saved to {output_file}")
            
        return html


def main():
    """Run the explainer with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explain user growth scores for crypto projects')
    parser.add_argument('--data', type=str, required=True, help='Path to data file (CSV or Excel)')
    parser.add_argument('--results', type=str, help='Path to results file (CSV)')
    parser.add_argument('--project', type=str, required=True, help='Project name to analyze')
    parser.add_argument('--viz', type=str, help='Path to save visualization (PNG)')
    parser.add_argument('--html', type=str, help='Path to save HTML explanation')
    
    args = parser.parse_args()
    
    # Load data
    try:
        if args.data.lower().endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.data)
        else:
            print(f"Unsupported file format: {args.data}")
            return
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    # Load results if available
    results_df = None
    if args.results:
        try:
            results_df = pd.read_csv(args.results)
        except Exception as e:
            print(f"Error loading results file: {e}")
    
    # Create explainer
    explainer = UserGrowthScoreExplainer(df, results_df)
    
    # Generate explanation
    explanation = explainer.get_user_growth_explanation(args.project)
    
    if 'error' in explanation:
        print(explanation['error'])
        return
    
    # Print summary
    print("\n" + "="*50)
    print(f"USER GROWTH ANALYSIS: {args.project}")
    print("="*50)
    
    if 'overall_score' in explanation and explanation['overall_score'] is not None:
        print(f"\nOverall Score: {explanation['overall_score']:.1f}/100")
        print(f"Growth Category: {explanation['growth_category']}")
    
    print(f"\nExplanation: {explanation['explanation']}")
    
    print("\nMetric Breakdown:")
    for metric, data in explanation.get('metrics', {}).items():
        if 'percentile' in data:
            print(f"  - {metric.replace('_', ' ').title()}: {data['percentile']:.1f}% percentile")
    
    # Create visualization if requested
    if args.viz:
        explainer.visualize_user_growth_explanation(args.project, args.viz)
    
    # Generate HTML if requested
    if args.html:
        explainer.generate_html_explanation(args.project, args.html)
        print(f"\nHTML explanation saved to {args.html}")


if __name__ == "__main__":
    main()