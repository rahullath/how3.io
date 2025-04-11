import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

# Import from earningsquality.py
from earnings_quality import (
    load_excel, 
    EarningsQualityAnalyzer, 
    calculate_revenue_magnitude, 
    calculate_revenue_quality
)

def format_number(value: float) -> str:
    """Format large numbers to be human-readable."""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

def extract_visualization_data(file_path: str, project_list: List[str]) -> Dict[str, Any]:
    """
    Extract simple visualization data for the specified projects.
    
    Args:
        file_path: Path to the Excel file with Token Terminal data
        project_list: List of projects to analyze
        
    Returns:
        Dictionary with visualization data
    """
    # Load the data
    df = load_excel(file_path)
    if df is None or df.empty:
        print(f"Error: Could not load data from {file_path}")
        return {}
    
    # Initialize EarningsQualityAnalyzer
    analyzer = EarningsQualityAnalyzer(df)
    
    # Find maximum revenue for magnitude calculation
    max_revenue_col = None
    for col in df.columns:
        if 'Revenue_30d sum' in str(col):
            max_revenue_col = col
            break
    
    if max_revenue_col:
        max_revenue_sum = df[max_revenue_col].max()
    else:
        # Fallback to a reasonable default
        max_revenue_sum = 1000000000
    
    print(f"Using maximum revenue value: {max_revenue_sum}")
    
    # Extract data for each project
    visualization_data = {}
    
    for project_name in project_list:
        print(f"Processing {project_name}...")
        
        # Find the project in the dataframe
        project_rows = df[df['Project'] == project_name]
        
        if len(project_rows) == 0:
            print(f"Project '{project_name}' not found. Skipping.")
            continue
        
        # Get the row index
        row_idx = project_rows.index[0]
        
        # Get project details
        project_row = project_rows.iloc[0]
        sector = project_row.get('Market sector', 'Unknown')
        
        # Calculate EQS metrics
        stability_score, stability_explanation = analyzer.calculate_stability_score(project_row)
        magnitude_score = calculate_revenue_magnitude(project_row, max_revenue_sum)
        eqs, eqs_explanation = calculate_revenue_quality(project_row, max_revenue_sum)
        
        # Format for easy visualization
        project_data = {
            "name": project_name,
            "sector": sector,
            "scores": {
                "earnings_quality": round(eqs, 2) if eqs is not None else None,
                "stability": round(stability_score, 2) if stability_score is not None else None,
                "magnitude": round(magnitude_score, 2) if magnitude_score is not None else None
            },
            "charts": {}
        }
        
        # Chart 1: EQS Score Gauge (simple gauge chart)
        # Shows the overall EQS score on a scale of 0-100
        if eqs is not None:
            project_data["charts"]["eqs_gauge"] = {
                "type": "gauge",
                "title": "Earnings Quality Score",
                "description": "Overall score based on revenue stability and magnitude",
                "data": {
                    "value": round(eqs, 1),
                    "min": 0,
                    "max": 100,
                    "ranges": [
                        {"min": 0, "max": 40, "label": "Poor", "color": "#FF5252"},
                        {"min": 40, "max": 70, "label": "Average", "color": "#FFC107"},
                        {"min": 70, "max": 100, "label": "Excellent", "color": "#4CAF50"}
                    ]
                }
            }
        
        # Chart 2: Stability vs Industry (simple comparison)
        # Shows how this project's stability compares to industry average
        if stability_score is not None:
            # Calculate industry average for stability
            industry_rows = df[df['Market sector'] == sector]
            industry_stability_scores = []
            
            for _, ind_row in industry_rows.iterrows():
                ind_stability, _ = analyzer.calculate_stability_score(ind_row)
                if ind_stability is not None:
                    industry_stability_scores.append(ind_stability)
            
            industry_avg = np.mean(industry_stability_scores) if industry_stability_scores else 50
            
            project_data["charts"]["stability_comparison"] = {
                "type": "comparison",
                "title": "Revenue Stability",
                "description": "How consistent is this project's revenue compared to others in its category",
                "data": {
                    "project": round(stability_score, 1),
                    "industry_average": round(industry_avg, 1),
                    "max": 100
                }
            }
        
        # Chart 3: Annualized Revenue (single stat with comparison)
        # Shows the project's annualized revenue and how it compares to sector average
        revenue_365d = None
        for col in df.columns:
            if 'Revenue_365d sum' in str(col) and col in project_row.index:
                revenue_365d = project_row[col]
                break
        
        if revenue_365d is not None and isinstance(revenue_365d, (int, float)) and revenue_365d > 0:
            # Calculate industry average
            industry_rows = df[df['Market sector'] == sector]
            industry_revenues = []
            
            for _, ind_row in industry_rows.iterrows():
                if col in ind_row.index and isinstance(ind_row[col], (int, float)) and ind_row[col] > 0:
                    industry_revenues.append(ind_row[col])
            
            industry_avg = np.mean(industry_revenues) if industry_revenues else None
            
            project_data["charts"]["annual_revenue"] = {
                "type": "stat",
                "title": "Annual Revenue",
                "description": "Total revenue generated over the past year",
                "data": {
                    "value": revenue_365d,
                    "formatted_value": format_number(revenue_365d),
                    "industry_average": industry_avg,
                    "formatted_industry_average": format_number(industry_avg) if industry_avg else "N/A",
                    "comparison": "above" if industry_avg and revenue_365d > industry_avg else "below"
                }
            }
        
        # Chart 4: Revenue by Time Period (bar chart) - now with proper ordering and formatting
        # Simple comparison of how consistent revenue is across different time periods
        periods_order = {'30d': 0, '90d': 1, '180d': 2, '365d': 3}  # Define order
        revenue_data = []
        for period in ['30d', '90d', '180d', '365d']:
            col_name = f'Revenue_{period} sum'
            if col_name in project_row.index and pd.notna(project_row[col_name]):
                value = project_row[col_name]
                if isinstance(value, (int, float)) and value > 0:
                    # For shorter periods, annualize the revenue for better comparison
                    annualized_value = value
                    if period == '30d':
                        annualized_value = value * 12  # 30d to annual
                        description = "Monthly revenue (annualized)"
                    elif period == '90d':
                        annualized_value = value * 4   # 90d to annual
                        description = "Quarterly revenue (annualized)"
                    elif period == '180d':
                        annualized_value = value * 2   # 180d to annual
                        description = "Semi-annual revenue (annualized)"
                    else:
                        description = "Annual revenue"
                    
                    revenue_data.append({
                        "period": period,
                        "period_name": {
                            '30d': '1 Month', 
                            '90d': '3 Months', 
                            '180d': '6 Months', 
                            '365d': '12 Months'
                        }[period],
                        "value": round(value, 2),
                        "formatted_value": format_number(value),
                        "annualized_value": round(annualized_value, 2),
                        "formatted_annualized": format_number(annualized_value),
                        "description": description,
                        "order": periods_order[period]
                    })
        
        if revenue_data:
            # Sort by the defined order
            revenue_data.sort(key=lambda x: x["order"])
            
            project_data["charts"]["revenue_consistency"] = {
                "type": "bar",
                "title": "Revenue Consistency Check",
                "description": "Revenue across different time periods (annualized to show consistency)",
                "data": revenue_data
            }
        
        # Store project data
        visualization_data[project_name] = project_data
    
    return visualization_data

def main():
    # List of projects to analyze
    projects = [
        "Aave",
        "GMX", 
        "Lido Finance",
        "Bitcoin",
        "Uniswap Labs",
        "BNB Chain",
        "OP Mainnet",
        "Pendle",
        "Curve",
        "Maple Finance",
        "Compound",
        "Ethereum",
        "Avalanche",
        "Chainlink"
    ]
    
    # Path to the data file
    file_path = "tt-master-data_2025-03-27.xlsx"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    # Extract visualization data
    viz_data = extract_visualization_data(file_path, projects)
    
    # Save to JSON file
    output_file = "eqs_visualization_data.json"
    with open(output_file, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"Visualization data saved to {output_file}")
    
    # Print summary for verification
    print("\nVisualization data summary:")
    for project, data in viz_data.items():
        print(f"{project}:")
        print(f"  EQS: {data['scores']['earnings_quality']}")
        print(f"  Charts: {list(data['charts'].keys())}")
    
    # Create a simple HTML preview of the charts
    create_html_preview(viz_data, "eqs_charts_preview.html")
    print(f"HTML preview saved to eqs_charts_preview.html")

def create_html_preview(viz_data: Dict[str, Any], output_file: str):
    """Create a simple HTML preview of the charts."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>EQS Visualization Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-gauge@0.3.0/dist/chartjs-gauge.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
        .project { margin-bottom: 40px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-container { width: 100%; margin-bottom: 30px; }
        .chart-row { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
        .chart-col { flex: 1; min-width: 300px; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 0; }
        h3 { color: #555; margin-top: 0; margin-bottom: 10px; }
        .description { color: #666; margin-bottom: 15px; font-size: 14px; }
        .stat-card { text-align: center; padding: 20px; }
        .stat-value { font-size: 28px; font-weight: bold; color: #4285F4; margin-bottom: 5px; }
        .stat-comparison { font-size: 14px; color: #666; }
        .gauge-container { height: 200px; }
        .comparison-container { padding: 20px; position: relative; height: 60px; }
        .comparison-bar { height: 20px; background-color: #e0e0e0; border-radius: 10px; position: relative; margin-bottom: 10px; }
        .project-bar { height: 100%; border-radius: 10px; background-color: #4285F4; position: absolute; }
        .industry-marker { width: 2px; height: 30px; background-color: #FF5252; position: absolute; transform: translateX(-50%); }
        .industry-label { position: absolute; color: #FF5252; font-size: 12px; transform: translateX(-50%); }
        .project-label { color: #4285F4; font-size: 12px; position: absolute; transform: translateX(-50%); }
        .bar-chart { height: 300px; }
    </style>
</head>
<body>
    <h1>EQS Visualization Preview</h1>
"""

    for project, data in viz_data.items():
        html += f"""
    <div class="project">
        <h2>{project} ({data['sector']})</h2>
        <p>EQS: {data['scores']['earnings_quality'] if data['scores']['earnings_quality'] is not None else 'N/A'} 
           (Stability: {data['scores']['stability'] if data['scores']['stability'] is not None else 'N/A'}, 
           Magnitude: {data['scores']['magnitude'] if data['scores']['magnitude'] is not None else 'N/A'})</p>
        
        <div class="chart-row">
"""

        # Add each chart
        charts_added = 0
        
        # EQS Gauge
        if 'eqs_gauge' in data['charts']:
            chart = data['charts']['eqs_gauge']
            gauge_id = f"{project.lower().replace(' ', '_')}_gauge"
            
            html += f"""
            <div class="chart-col">
                <h3>{chart['title']}</h3>
                <p class="description">{chart['description']}</p>
                <div class="gauge-container">
                    <canvas id="{gauge_id}"></canvas>
                </div>
                <script>
                    (function() {{
                        const ctx = document.getElementById('{gauge_id}').getContext('2d');
                        const gauge = new Chart(ctx, {{
                            type: 'gauge',
                            data: {{
                                datasets: [{{
                                    value: {chart['data']['value']},
                                    minValue: {chart['data']['min']},
                                    maxValue: {chart['data']['max']},
                                    backgroundColor: ['#FF5252', '#FFC107', '#4CAF50'],
                                    data: [40, 70, 100]
                                }}]
                            }},
                            options: {{
                                needle: {{
                                    radiusPercentage: 2,
                                    widthPercentage: 3.2,
                                    lengthPercentage: 80,
                                    color: 'rgba(0, 0, 0, 1)'
                                }},
                                valueLabel: {{
                                    display: true,
                                    formatter: function(value) {{ return value.toFixed(1); }},
                                    color: '#444',
                                    backgroundColor: 'rgba(0, 0, 0, 0)',
                                    borderRadius: 5,
                                    padding: {{
                                        top: 10,
                                        bottom: 10
                                    }}
                                }}
                            }}
                        }});
                    }})();
                </script>
            </div>
"""
            charts_added += 1
        
        # Stability Comparison
        if 'stability_comparison' in data['charts']:
            chart = data['charts']['stability_comparison']
            comparison_id = f"{project.lower().replace(' ', '_')}_stability"
            
            project_value = chart['data']['project']
            industry_value = chart['data']['industry_average']
            max_value = chart['data']['max']
            
            project_percent = (project_value / max_value) * 100
            industry_percent = (industry_value / max_value) * 100
            
            html += f"""
            <div class="chart-col">
                <h3>{chart['title']}</h3>
                <p class="description">{chart['description']}</p>
                <div class="comparison-container">
                    <div class="comparison-bar">
                        <div class="project-bar" style="width: {project_percent}%;"></div>
                        <div class="industry-marker" style="left: {industry_percent}%;"></div>
                    </div>
                    <div class="project-label" style="left: {project_percent}%;">{project_value}</div>
                    <div class="industry-label" style="left: {industry_percent}%;">Industry: {industry_value}</div>
                </div>
            </div>
"""
            charts_added += 1
        
        # Close chart row if needed and start a new one
        if charts_added % 2 == 0:
            html += """
        </div>
        <div class="chart-row">
"""
        
        # Annual Revenue Stat
        if 'annual_revenue' in data['charts']:
            chart = data['charts']['annual_revenue']
            
            html += f"""
            <div class="chart-col">
                <h3>{chart['title']}</h3>
                <p class="description">{chart['description']}</p>
                <div class="stat-card">
                    <div class="stat-value">{chart['data']['formatted_value']}</div>
                    <div class="stat-comparison">Industry Average: {chart['data']['formatted_industry_average']}</div>
                </div>
            </div>
"""
            charts_added += 1
        
        # Close chart row if needed and start a new one
        if charts_added % 2 == 0:
            html += """
        </div>
        <div class="chart-row">
"""
        
        # Revenue Consistency
        if 'revenue_consistency' in data['charts']:
            chart = data['charts']['revenue_consistency']
            chart_id = f"{project.lower().replace(' ', '_')}_revenue_consistency"
            
            labels = [item['period_name'] for item in chart['data']]
            values = [item['annualized_value'] for item in chart['data']]
            tooltips = [item['description'] + ": " + item['formatted_annualized'] for item in chart['data']]
            
            html += f"""
            <div class="chart-col" style="flex: 2;">
                <h3>{chart['title']}</h3>
                <p class="description">{chart['description']}</p>
                <div class="bar-chart">
                    <canvas id="{chart_id}"></canvas>
                </div>
                <script>
                    (function() {{
                        const ctx = document.getElementById('{chart_id}').getContext('2d');
                        new Chart(ctx, {{
                            type: 'bar',
                            data: {{
                                labels: {json.dumps(labels)},
                                datasets: [{{
                                    label: 'Annualized Revenue',
                                    data: {json.dumps(values)},
                                    backgroundColor: '#4285F4',
                                    barPercentage: 0.6
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    tooltip: {{
                                        callbacks: {{
                                            label: function(context) {{
                                                return {json.dumps(tooltips)}[context.dataIndex];
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: true,
                                        ticks: {{
                                            callback: function(value) {{
                                                if (value >= 1e9) return '$' + (value / 1e9).toFixed(1) + 'B';
                                                if (value >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
                                                if (value >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
                                                return '$' + value;
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }})();
                </script>
            </div>
"""
            charts_added += 1
        
        # Close the chart row
        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()