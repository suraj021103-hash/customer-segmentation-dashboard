# Customer Segregation with K-Means

This project segments mall customers using K-Means clustering on the `Mall_Customers.csv` dataset.

## Files
- `Mall_Customers.csv` - source dataset
- `customer_segmentation.py` - analysis script
- `requirements.txt` - Python dependencies

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run segmentation:
   ```bash
   python customer_segmentation.py
   ```

## Output
- `elbow_plot.png` - elbow method chart for choosing `k`
- `customer_clusters.png` - cluster scatter plot
- `customer_segments.csv` - dataset with assigned cluster labels
