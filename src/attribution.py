"""Attribution Modeling Module.

This module provides various attribution models for
understanding customer journey touchpoints.
"""

import pandas as pd
import numpy as np
from typing import Any


class AttributionModeler:
    """Handles marketing attribution modeling."""

    def __init__(self):
        """Initialize the AttributionModeler."""
        self.models = {}

    def first_touch_attribution(
        self,
        customer_journeys: pd.DataFrame,
        touchpoint_col: str,
        conversion_col: str
    ) -> pd.Series:
        """Attribute conversion to first touchpoint.

        Args:
            customer_journeys: DataFrame with customer journeys.
            touchpoint_col: Column with touchpoints.
            conversion_col: Conversion indicator column.

        Returns:
            Attribution weights per touchpoint.
        """
        first_touches = customer_journeys.groupby("customer_id")[touchpoint_col].first()
        converted = customer_journeys[customer_journeys[conversion_col] == 1]

        attribution = converted.groupby("customer_id")[touchpoint_col].first()
        attribution_counts = attribution.value_counts()

        return attribution_counts

    def last_touch_attribution(
        self,
        customer_journeys: pd.DataFrame,
        touchpoint_col: str,
        conversion_col: str
    ) -> pd.Series:
        """Attribute conversion to last touchpoint.

        Args:
            customer_journeys: DataFrame with customer journeys.
            touchpoint_col: Column with touchpoints.
            conversion_col: Conversion indicator column.

        Returns:
            Attribution weights per touchpoint.
        """
        converted = customer_journeys[customer_journeys[conversion_col] == 1]
        attribution = converted.groupby("customer_id")[touchpoint_col].last()

        return attribution.value_counts()

    def linear_attribution(
        self,
        customer_journeys: pd.DataFrame,
        touchpoint_col: str,
        conversion_col: str
    ) -> pd.Series:
        """Apply linear attribution across all touchpoints.

        Args:
            customer_journeys: DataFrame with customer journeys.
            touchpoint_col: Column with touchpoints.
            conversion_col: Conversion indicator column.

        Returns:
            Attribution weights per touchpoint.
        """
        converted = customer_journeys[customer_journeys[conversion_col] == 1]

        touchpoint_counts = converted.groupby("customer_id")[touchpoint_col].count()
        attribution_weights = pd.Series(dtype=float)

        for customer_id, group in converted.groupby("customer_id"):
            weight = 1.0 / len(group)
            for touchpoint in group[touchpoint_col]:
                attribution_weights[touchpoint] = attribution_weights.get(touchpoint, 0) + weight

        return attribution_weights

    def position_based_attribution(
        self,
        customer_journeys: pd.DataFrame,
        touchpoint_col: str,
        conversion_col: str,
        first_weight: float = 0.4,
        last_weight: float = 0.4
    ) -> pd.Series:
        """Apply position-based (U-shaped) attribution.

        Args:
            customer_journeys: DataFrame with customer journeys.
            touchpoint_col: Column with touchpoints.
            conversion_col: Conversion indicator column.
            first_weight: Weight for first touchpoint.
            last_weight: Weight for last touchpoint.

        Returns:
            Attribution weights per touchpoint.
        """
        converted = customer_journeys[customer_journeys[conversion_col] == 1]
        attribution_weights = pd.Series(dtype=float)

        for customer_id, group in converted.groupby("customer_id"):
            touchpoints = group[touchpoint_col].tolist()
            n_touchpoints = len(touchpoints)

            if n_touchpoints == 1:
                attribution_weights[touchpoints[0]] = attribution_weights.get(touchpoints[0], 0) + 1.0
            else:
                attribution_weights[touchpoints[0]] = attribution_weights.get(touchpoints[0], 0) + first_weight
                attribution_weights[touchpoints[-1]] = attribution_weights.get(touchpoints[-1], 0) + last_weight

                middle_weight = (1 - first_weight - last_weight) / (n_touchpoints - 2)
                for tp in touchpoints[1:-1]:
                    attribution_weights[tp] = attribution_weights.get(tp, 0) + middle_weight

        return attribution_weights

    def calculate_roi(
        self,
        channel_costs: dict[str, float],
        channel_conversions: dict[str, int],
        average_order_value: float
    ) -> pd.DataFrame:
        """Calculate ROI per channel.

        Args:
            channel_costs: Dictionary of costs per channel.
            channel_conversions: Dictionary of conversions per channel.
            average_order_value: Average order value.

        Returns:
            DataFrame with ROI metrics.
        """
        data = []
        for channel in channel_costs:
            cost = channel_costs[channel]
            conversions = channel_conversions.get(channel, 0)
            revenue = conversions * average_order_value
            roi = (revenue - cost) / cost if cost > 0 else 0

            data.append({
                "channel": channel,
                "cost": cost,
                "conversions": conversions,
                "revenue": revenue,
                "roi": roi
            })

        return pd.DataFrame(data)


def simulate_markov_attribution(
    customer_journeys: pd.DataFrame,
    touchpoint_col: str,
    conversion_col: str
) -> dict[str, float]:
    """Calculate attribution using Markov chain model.

    Args:
        customer_journeys: DataFrame with customer journeys.
        touchpoint_col: Column with touchpoints.
        conversion_col: Conversion indicator column.

    Returns:
        Removal effect per channel.
    """
    all_touchpoints = customer_journeys[touchpoint_col].unique()
    removal_effects = {}

    for channel in all_touchpoints:
        removed = customer_journeys[customer_journeys[touchpoint_col] != channel]
        original_conversions = customer_journeys[conversion_col].sum()

        if len(removed) > 0:
            removed_conversions = removed[conversion_col].sum()
            removal_effect = 1 - (removed_conversions / original_conversions) if original_conversions > 0 else 0
        else:
            removal_effect = 1.0

        removal_effects[str(channel)] = removal_effect

    return removal_effects
