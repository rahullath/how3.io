import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import sys

def export_results_to_csv(explanations: List[Dict[str, Any]], df: pd.DataFrame, csv_path: str):
    """
    Export the comprehensive scores, explanations, and the original data to a CSV file.

    Args:
        explanations: List of dictionaries containing explanations for each project.
        df: The original DataFrame with the comprehensive scores added.
        csv_path: The path to the CSV file where the results will be saved.
    """
    # Create a list to hold the data for the CSV
    data = []

    # Iterate through each row in the DataFrame and its corresponding explanation
    for index, row in df.iterrows():
        # Get the explanation for the current row
        explanation = explanations[index]

        # Extract the comprehensive score
        comprehensive_score = explanation["comprehensive_score"]

        # Extract individual scores and methods
        stability_score = explanation["stability"]["stability_score"]
        stability_method = explanation["stability"]["method"]
        revenue_diversification_score = explanation["revenue_diversification"]["diversification_score"]
        revenue_diversification_method = explanation["revenue_diversification"]["method_used"]
        user_efficiency_score = explanation["user_efficiency"]["score"]
        user_efficiency_method = explanation["user_efficiency"]["method"]
        sustainability_score = explanation["sustainability"]["score"]
        sustainability_method = explanation["sustainability"]["method"]
        transaction_activity_score = explanation["transaction_activity"]["score"]
        transaction_activity_method = explanation["transaction_activity"]["method"]

        # Create a dictionary for the current row
        row_data = {
            "Project": row["Project"],
            "Market sector": row["Market sector"],
            "Comprehensive Score": comprehensive_score,
            "Stability Score": stability_score,
            "Stability Method": stability_method,
            "Revenue Diversification Score": revenue_diversification_score,
            "Revenue Diversification Method": revenue_diversification_method,
            "User Efficiency Score": user_efficiency_score,
            "User Efficiency Method": user_efficiency_method,
            "Sustainability Score": sustainability_score,
            "Sustainability Method": sustainability_method,
            "Transaction Activity Score": transaction_activity_score,
            "Transaction Activity Method": transaction_activity_method,
        }

        # Add the row data to the list
        data.append(row_data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_path, index=False)
    print(f"Results exported to {csv_path}")
