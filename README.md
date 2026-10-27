# ASA Data Quest 2026
## Global Demographic Clustering Analysis

This repository contains the Python scripts developed for **ASA Data Quest 2026**. The project explores global demographic patterns by analyzing fertility rates, life expectancy, and changes observed between 1990 and 2021.

Using clustering and dimensionality reduction, the analysis groups countries according to demographic characteristics and evaluates how stable those groups are under different modeling choices.

## Research Objectives

- Identify demographic patterns among countries using K-Means clustering.
- Examine changes in fertility rates and life expectancy over time.
- Visualize demographic clusters using Principal Component Analysis (PCA).
- Evaluate clustering quality using the Silhouette Score.
- Compare clustering methods using the Adjusted Rand Index (ARI).
- Investigate how removing the COVID-related life-expectancy indicator affects cluster assignments.

## Data Sources

The analysis uses demographic indicators from the United Nations *World Population Prospects 2024* datasets.

The scripts expect the following files:

- `WPP2024_Demographic_Indicators_Medium_csv.gz`
- `WPP2024_Supplement_Births_and_LE60.csv`

**Dataset availability:** The original datasets are not included in this repository. Users must obtain the files through appropriate authorized sources and comply with the applicable data-use terms and ethical requirements.

## Methodology

### 1. Data Preparation
The scripts select country-level records, combine the relevant years, calculate demographic indicators, and remove records with missing values where required.

### 2. Feature Engineering
The analysis uses indicators including:

- **TFR_2019:** Total fertility rate in 2019.
- **TFR_Change:** Change in total fertility rate between 1990 and 2019.
- **TFR_Change_Pct:** Percentage change in total fertility rate between 1990 and 2019.
- **LE60_2019:** Remaining life expectancy at age 60 in 2019.
- **LE60_Change:** Change in remaining life expectancy at age 60 between 1990 and 2019.
- **Covid_Shock:** Difference between remaining life expectancy at age 60 in 2019 and 2021.

The exact features vary between scripts.

### 3. Clustering and Visualization
K-Means clustering is used to group countries into three clusters. StandardScaler standardizes the features, and PCA reduces the feature space to two dimensions for visualization.

### 4. Model Evaluation
The project uses the Silhouette Score to assess cluster separation and the Adjusted Rand Index to compare clustering assignments across different methods and random seeds.

## Repository Structure

| File | Purpose |
|---|---|
| `old_data_analysis_1.py` | Performs K-Means clustering, calculates the Silhouette Score, labels demographic clusters, and visualizes the results using PCA. |
| `new_data_analyze_2.py` | Extends the analysis with missing-data reporting, percentage fertility change, a correlation matrix, PCA loadings, and representative countries near cluster centroids. |
| `Silhoutte_Score_1.py` | Evaluates K-Means clustering for different values of *k* (2–6) using the Silhouette Score. |
| `ARI_table.py` | Compares clustering methods, evaluates cluster stability using ARI, tests sensitivity to removing the COVID-related feature, and generates a comparison table. |
| `.gitignore` | Specifies files and directories Git should ignore. |
| `LICENSE` | Contains the project's license. |

## Requirements

The project uses Python and the following libraries:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SciPy

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to Run

1. Install Python and the required libraries.
2. Obtain the required datasets from an authorized source.
3. Place the datasets in a local data directory.
4. Open the Python scripts and update their data-directory settings to match your computer.
5. Run the desired script from your terminal or Python environment.

For example:

```bash
python new_data_analyze_2.py
```

The scripts currently use local directory settings, so these paths may need to be adjusted before execution. `ARI_table.py` supports configuring its data and output directories through the `DATA_DIR` and `OUT_DIR` environment variables.

Some scripts display plots, while `ARI_table.py` also saves a generated image named `Slayt2_Tablo_ve_Covid.png` to its configured output directory.

## Important Notes

- Cluster labels describe groups identified by the model; they should not be interpreted as definitive classifications of countries.
- The `Covid_Shock` variable measures the difference between the 2019 and 2021 values. It is not, by itself, proof that COVID-19 caused the entire difference.
- Results depend on the selected features, available observations, preprocessing, and clustering settings.

## License

See the [`LICENSE`](LICENSE) file for the terms under which this project is distributed.

## Acknowledgements

This project was developed for **ASA Data Quest 2026** and uses demographic indicators from United Nations World Population Prospects 2024.
