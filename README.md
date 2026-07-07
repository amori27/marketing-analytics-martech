# Marketing Analytics

Customer segmentation via scikit-learn, A/B test significance testing with SciPy, multi-touch attribution models, and campaign reporting with Seaborn/Matplotlib.

## Usage

```python
from src.segmentation import CustomerSegmentation
segmenter = CustomerSegmentation()
segments = segmenter.segment_customers(customer_data)
```

```python
from src.ab_testing import ABTestAnalyzer
analyzer = ABTestAnalyzer()
result = analyzer.analyze(test_group, control_group)
```

## Project Structure

```
src/
├── segmentation.py     # Customer segmentation
├── ab_testing.py       # A/B test analysis
├── attribution.py      # Attribution modeling
└── reporting.py        # Analytics reporting
```

## License

MIT
