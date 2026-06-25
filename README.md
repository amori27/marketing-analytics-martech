# Marketing Analytics & MarTech
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Comprehensive marketing analytics toolkit featuring customer segmentation, A/B testing analysis, attribution modeling, and campaign performance tracking.

## Description

Production-ready marketing analytics platform for analyzing campaign performance, customer behavior, and ROI. Implements statistical testing, cohort analysis, and predictive modeling for data-driven marketing decisions.

## Skills & Technologies

- Python 3.9+
- Pandas & NumPy
- SciPy (Statistical Analysis)
- Seaborn & Matplotlib
- Scikit-learn (Predictive Models)
- A/B Testing
- Customer Segmentation
- Attribution Modeling

## Installation

```bash
git clone https://github.com/amori27/marketing-analytics-martech.git
cd marketing-analytics-martech
pip install -r requirements.txt
```

## Usage

### Customer Segmentation

```python
from src.segmentation import CustomerSegmentation

segmenter = CustomerSegmentation()
segments = segmenter.segment_customers(customer_data)
```

### A/B Testing

```python
from src.ab_testing import ABTestAnalyzer

analyzer = ABTestAnalyzer()
result = analyzer.analyze(test_group, control_group)
```

## Project Structure

```
marketing-analytics-martech/
├── src/
│   ├── segmentation.py     # Customer segmentation
│   ├── ab_testing.py       # A/B test analysis
│   ├── attribution.py       # Attribution modeling
│   └── reporting.py        # Analytics reporting
├── requirements.txt
└── README.md
```

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [A/B Testing Guide](https://experimentguide.com/)

## License

MIT License
