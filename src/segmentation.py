"""Customer Segmentation Module.

This module provides customer segmentation using clustering
algorithms and RFM (Recency, Frequency, Monetary) analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Any


class CustomerSegmentation:
    """Handles customer segmentation using clustering."""

    def __init__(self, n_clusters: int = 5):
        """Initialize the CustomerSegmentation.

        Args:
            n_clusters: Number of segments/clusters.
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.segment_labels = ["Bronze", "Silver", "Gold", "Platinum", "VIP"]

    def calculate_rfm(
        self,
        transactions: pd.DataFrame,
        customer_id_col: str,
        transaction_date_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) metrics.

        Args:
            transactions: Transaction data.
            customer_id_col: Customer ID column name.
            transaction_date_col: Transaction date column name.
            amount_col: Transaction amount column name.

        Returns:
            DataFrame with RFM metrics.
        """
        reference_date = transactions[transaction_date_col].max()

        rfm = transactions.groupby(customer_id_col).agg({
            transaction_date_col: lambda x: (reference_date - x.max()).days,
            customer_id_col: "count",
            amount_col: "sum"
        }).rename(columns={
            transaction_date_col: "Recency",
            customer_id_col: "Frequency",
            amount_col: "Monetary"
        })

        rfm = rfm.reset_index()
        return rfm

    def segment_customers(self, rfm_data: pd.DataFrame) -> pd.DataFrame:
        """Perform customer segmentation using K-Means.

        Args:
            rfm_data: DataFrame with RFM metrics.

        Returns:
            DataFrame with segment labels.
        """
        features = rfm_data[["Recency", "Frequency", "Monetary"]].copy()

        features_scaled = self.scaler.fit_transform(features)

        rfm_data["Segment"] = self.model.fit_predict(features_scaled)

        segment_mapping = {
            i: self.segment_labels[i % len(self.segment_labels)]
            for i in range(self.n_clusters)
        }
        rfm_data["Segment_Name"] = rfm_data["Segment"].map(segment_mapping)

        return rfm_data

    def get_segment_summary(self, segmented_data: pd.DataFrame) -> pd.DataFrame:
        """Generate segment summary statistics.

        Args:
            segmented_data: DataFrame with segment labels.

        Returns:
            Summary statistics per segment.
        """
        summary = segmented_data.groupby("Segment_Name").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": ["mean", "sum", "count"]
        }).round(2)

        return summary

    def calculate_segment_value(self, segmented_data: pd.DataFrame) -> dict[str, float]:
        """Calculate total value per segment.

        Args:
            segmented_data: DataFrame with segment labels.

        Returns:
            Dictionary of segment values.
        """
        segment_values = segmented_data.groupby("Segment_Name")["Monetary"].sum()
        return segment_values.to_dict()


def create_customer_persona(segment_data: dict[str, Any]) -> dict[str, str]:
    """Create a customer persona description from segment data.

    Args:
        segment_data: Segment statistics.

    Returns:
        Persona description.
    """
    persona = {
        "name": "Customer Persona",
        "recency_score": "High" if segment_data.get("Recency", 0) < 30 else "Low",
        "frequency_score": "High" if segment_data.get("Frequency", 0) > 10 else "Low",
        "monetary_score": "High" if segment_data.get("Monetary", 0) > 1000 else "Low"
    }

    if persona["monetary_score"] == "High" and persona["frequency_score"] == "High":
        persona["recommended_strategy"] = "VIP treatment, exclusive offers"
    elif persona["frequency_score"] == "High":
        persona["recommended_strategy"] = "Loyalty rewards, upselling"
    elif persona["recency_score"] == "Low":
        persona["recommended_strategy"] = "Win-back campaigns, re-engagement"
    else:
        persona["recommended_strategy"] = "Nurture with educational content"

    return persona
