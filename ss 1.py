import gzip
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
os.chdir(r"C:\Work")

df_wpp = pd.read_csv("WPP2024_Demographic_Indicators_Medium_csv.gz", low_memory=False)
df_le60 = pd.read_csv("WPP2024_Supplement_Births_and_LE60.csv", low_memory=False)

df_wpp = df_wpp[df_wpp["LocTypeName"] == "Country/Area"]

tfr_1990 = df_wpp[df_wpp["Time"] == 1990][["Location", "TFR"]].rename(
    columns={"TFR": "TFR_1990"}
)
tfr_2019 = df_wpp[df_wpp["Time"] == 2019][["Location", "TFR"]].rename(
    columns={"TFR": "TFR_2019"}
)

le_1990 = df_le60[df_le60["Time"] == 1990][["Location", "LE60"]].rename(
    columns={"LE60": "LE60_1990"}
)
le_2019 = df_le60[df_le60["Time"] == 2019][["Location", "LE60"]].rename(
    columns={"LE60": "LE60_2019"}
)
le_2021 = df_le60[df_le60["Time"] == 2021][["Location", "LE60"]].rename(
    columns={"LE60": "LE60_2021"}
)

analiz_verisi = tfr_1990.merge(tfr_2019, on="Location", how="inner")
for df in [le_1990, le_2019, le_2021]:
  analiz_verisi = analiz_verisi.merge(df, on="Location", how="inner")

analiz_verisi["TFR_Change"] = (
    analiz_verisi["TFR_2019"] - analiz_verisi["TFR_1990"]
)
analiz_verisi["LE60_Change"] = (
    analiz_verisi["LE60_2019"] - analiz_verisi["LE60_1990"]
)
analiz_verisi["Covid_Shock"] = (
    analiz_verisi["LE60_2019"] - analiz_verisi["LE60_2021"]
)

df_clean = analiz_verisi.dropna().copy()
df_clean.set_index("Location", inplace=True)

features = ["TFR_2019", "TFR_Change", "LE60_2019", "LE60_Change", "Covid_Shock"]
X = df_clean[features]

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

for k in range(2, 7):
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels = kmeans.fit_predict(X_std)
  score = silhouette_score(X_std, labels)
  print(f"k={k}: Silhouette Skoru = {score:.4f}")