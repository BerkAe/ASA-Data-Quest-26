# ASA Data Quest 2026
### Global Demographic Analysis

This project explores global demographic patterns using fertility rates and life expectancy data from the United Nations World Population Prospects 2024.

Using Python, it groups countries with similar demographic characteristics and examines how these patterns change over time.

## What We Did

- Used **K-Means clustering** to group countries.
- Used **PCA** to visualize demographic patterns.
- Evaluated clustering with the **Silhouette Score** and **Adjusted Rand Index (ARI)**.
- Compared models and tested the effect of excluding a COVID-related indicator.

## Project Files

| File | Description |
|---|---|
| `old_data_analysis_1.py` | Main clustering analysis and visualization. |
| `new_data_analyze_2.py` | Extended analysis of demographic indicators and clusters. |
| `Silhoutte_Score_1.py` | Tests different numbers of clusters. |
| `ARI_table.py` | Compares clustering methods and tests model stability. |

## How to Use

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Download the required datasets from an authorized source, place them in your local data directory, and update the file paths in the scripts before running them.

## Data Availability

The original datasets are not included in this repository. Please obtain them through authorized sources and follow their applicable usage terms.

## License

See the [LICENSE](LICENSE) file.

---

*Developed for ASA Data Quest 2026.*
