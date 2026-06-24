"""A/B Testing Analysis Module.

This module provides statistical analysis for A/B testing
including hypothesis testing, confidence intervals, and effect size calculations.
"""

import numpy as np
from scipy import stats
from typing import Any


class ABTestAnalyzer:
    """Handles A/B test statistical analysis."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the ABTestAnalyzer.

        Args:
            confidence_level: Statistical confidence level.
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_statistics(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, Any]:
        """Calculate descriptive statistics for both groups.

        Args:
            control: Control group data.
            treatment: Treatment group data.

        Returns:
            Dictionary of statistics.
        """
        return {
            "control": {
                "n": len(control),
                "mean": float(np.mean(control)),
                "std": float(np.std(control, ddof=1)),
                "median": float(np.median(control))
            },
            "treatment": {
                "n": len(treatment),
                "mean": float(np.mean(treatment)),
                "std": float(np.std(treatment, ddof=1)),
                "median": float(np.median(treatment))
            }
        }

    def two_sample_ttest(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        equal_variance: bool = True
    ) -> dict[str, Any]:
        """Perform two-sample t-test.

        Args:
            control: Control group data.
            treatment: Treatment group data.
            equal_variance: Assume equal variances.

        Returns:
            T-test results.
        """
        if equal_variance:
            statistic, pvalue = stats.ttest_ind(control, treatment)
        else:
            statistic, pvalue = stats.ttest_ind(control, treatment, equal_var=False)

        significant = pvalue < self.alpha

        return {
            "test": "two_sample_ttest",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": significant,
            "alpha": self.alpha
        }

    def mann_whitney_test(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric).

        Args:
            control: Control group data.
            treatment: Treatment group data.

        Returns:
            Mann-Whitney test results.
        """
        statistic, pvalue = stats.mannwhitneyu(control, treatment, alternative="two-sided")

        return {
            "test": "mann_whitney_u",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": pvalue < self.alpha,
            "alpha": self.alpha
        }

    def calculate_effect_size(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, float]:
        """Calculate Cohen's d effect size.

        Args:
            control: Control group data.
            treatment: Treatment group data.

        Returns:
            Effect size metrics.
        """
        mean_diff = np.mean(treatment) - np.mean(control)
        pooled_std = np.sqrt(
            ((len(control) - 1) * np.var(control, ddof=1) +
             (len(treatment) - 1) * np.var(treatment, ddof=1)) /
            (len(control) + len(treatment) - 2)
        )

        cohens_d = mean_diff / pooled_std

        return {
            "mean_difference": float(mean_diff),
            "cohens_d": float(cohens_d),
            "interpretation": self._interpret_effect_size(cohens_d)
        }

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            cohens_d: Cohen's d value.

        Returns:
            Interpretation string.
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def confidence_interval(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Calculate confidence interval for the difference.

        Args:
            control: Control group data.
            treatment: Treatment group data.

        Returns:
            Confidence intervals for both groups and difference.
        """
        control_ci = stats.t.interval(
            self.confidence_level,
            len(control) - 1,
            loc=np.mean(control),
            scale=stats.sem(control)
        )

        treatment_ci = stats.t.interval(
            self.confidence_level,
            len(treatment) - 1,
            loc=np.mean(treatment),
            scale=stats.sem(treatment)
        )

        diff_mean = np.mean(treatment) - np.mean(control)
        diff_se = np.sqrt(
            np.var(control, ddof=1) / len(control) +
            np.var(treatment, ddof=1) / len(treatment)
        )
        diff_ci = (diff_mean - 1.96 * diff_se, diff_mean + 1.96 * diff_se)

        return {
            "control_ci": (float(control_ci[0]), float(control_ci[1])),
            "treatment_ci": (float(treatment_ci[0]), float(treatment_ci[1])),
            "difference_ci": (float(diff_ci[0]), float(diff_ci[1]))
        }

    def analyze(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> dict[str, Any]:
        """Perform complete A/B test analysis.

        Args:
            control: Control group data.
            treatment: Treatment group data.

        Returns:
            Complete analysis results.
        """
        return {
            "statistics": self.calculate_statistics(control, treatment),
            "t_test": self.two_sample_ttest(control, treatment),
            "mann_whitney": self.mann_whitney_test(control, treatment),
            "effect_size": self.calculate_effect_size(control, treatment),
            "confidence_intervals": self.confidence_interval(control, treatment)
        }


def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """Calculate required sample size for A/B test.

    Args:
        baseline_rate: Baseline conversion rate.
        minimum_detectable_effect: Minimum effect to detect.
        alpha: Significance level.
        power: Statistical power.

    Returns:
        Required sample size per group.
    """
    from scipy.stats import norm

    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    pooled_p = (p1 + p2) / 2
    effect_size = abs(p2 - p1) / np.sqrt(pooled_p * (1 - pooled_p))

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))
