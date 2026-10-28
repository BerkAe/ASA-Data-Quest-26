# ASA Data Quest 2026: Global Demographic Transitions

This repository contains the code and methodology for the ASA Data Quest 2026 project. The project explores how countries group together based on their demographic trajectories, specifically focusing on fertility rates (TFR) and life expectancy (LE60) using the United Nations World Population Prospects (WPP) 2024 dataset.

The main objective is not just to cluster countries once, but to test the **stability of these clusters**. We analyzed how much the cluster boundaries shift when using different algorithms and measured the specific impact of removing the COVID-19 pandemic shock from the data.

---

## Approach

* **Data Preparation:** We calculated the percentage change in fertility rates and the absolute changes in life expectancy from 1990 to 2019 to ensure accurate comparisons across countries.
* **Clustering & Visualization:** Countries were grouped into distinct clusters using K-Means. We utilized PCA to visualize these clusters, mapping countries on a continuous demographic spectrum.
* **Strict Alignment:** To accurately compare different models, we used the Hungarian Algorithm to mathematically align our clusters, preventing the model from randomly swapping group labels.
* **Sensitivity Analysis:** We tested the K-Means results against alternative algorithms (GMM, Ward) and calculated the Adjusted Rand Index (ARI). This allowed us to quantify exactly how many countries change groups based on methodology or the exclusion of the COVID-19 shock.

---

## Repository Contents

* **`final_run.py`** – Core Script. This production-ready script executes the entire pipeline, handles missing data, aligns the clusters, and generates all the CSV reports used in the final presentation. 
* **`ARI_table.py`** – Generates the visual table comparing the clustering models and highlighting the impact of the COVID-19 shock.
* **`new_data_analyze_2.py`** – Creates the 2D PCA scatter plot of the clusters and automatically annotates the best representative countries for each group.
* **`Silhoutte_Score_1.py`** – The script used to mathematically validate that 3 clusters was the optimal choice.
* **`old_data_analysis_1.py`** – Early exploratory analysis script, preserved for version history.

---

## Execution Guide

### Requirements
Ensure Python 3.8+ is installed. Install the necessary libraries using:

    pip install pandas numpy matplotlib seaborn scikit-learn scipy

### Data Access
To comply with the ASA Ethical Guidelines and the UN's data redistribution policies, the raw CSV files are not included in this repository. 

* Download WPP2024_Demographic_Indicators_Medium_csv.gz and WPP2024_Supplement_Births_and_LE60.csv directly from the UN DESA Population Division website.
* Place these files in a designated local directory.

### Running the Analysis
Set the environment variable pointing to your data directory and run the main script:

For Windows (Command Prompt):
    set DATA_DIR=C:\Path\To\Your\Data
    python final_run.py

For Mac/Linux:
    export DATA_DIR=/Path/To/Your/Data
    python final_run.py

Upon completion, a final_run_outputs/ directory will be created containing all the generated reports.

---

## 📜 License
See the LICENSE file for details. Developed for the ASA Data Quest 2026.
