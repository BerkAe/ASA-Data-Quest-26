import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
os.chdir(r"C:\Work\Asa Data quest 26")

# Değişkenler:
# LE60 = Life Expectancy at age 60
# TFR = Total Fertility Rate (doğurganlık oranı)
# TFR_2019 --> 2019 Toplam Doğurganlık Oranı (Kadın başına ortalama çocuk sayısı)
# TFR_Change --> 1990-2019 arası TFR değişimi (TFR_2019 - TFR_1990)
# LE60_2019 --> 2019'da 60 yaşındaki kişinin kalan ortalama ömrü (Yıl)
# LE60_Change --> 1990-2019 arası LE60 değişimi (LE60_2019 - LE60_1990)
# Covid_Shock --> COVID-19 kaynaklı LE60 kaybı (LE60_2019 - LE60_2021)

#veri alma
df_wpp = pd.read_csv("WPP2024_Demographic_Indicators_Medium_csv.gz", low_memory=False)
df_le60 = pd.read_csv("WPP2024_Supplement_Births_and_LE60.csv", low_memory=False)

#filtreleme
df_wpp = df_wpp[df_wpp['LocTypeName'] == 'Country/Area']

# daha fazla veri
tfr_1990 = df_wpp[df_wpp['Time'] == 1990][['Location', 'TFR']].rename(columns={'TFR': 'TFR_1990'})
tfr_2019 = df_wpp[df_wpp['Time'] == 2019][['Location', 'TFR']].rename(columns={'TFR': 'TFR_2019'})

le_1990 = df_le60[df_le60['Time'] == 1990][['Location', 'LE60']].rename(columns={'LE60': 'LE60_1990'})
le_2019 = df_le60[df_le60['Time'] == 2019][['Location', 'LE60']].rename(columns={'LE60': 'LE60_2019'})
le_2021 = df_le60[df_le60['Time'] == 2021][['Location', 'LE60']].rename(columns={'LE60': 'LE60_2021'})

# silent loss engelleme
tum_ulkeler = pd.DataFrame({'Location': df_wpp['Location'].unique()})
analiz_verisi = tum_ulkeler.merge(tfr_1990, on='Location', how='left') \
                           .merge(tfr_2019, on='Location', how='left') \
                           .merge(le_1990, on='Location', how='left') \
                           .merge(le_2019, on='Location', how='left') \
                           .merge(le_2021, on='Location', how='left')

# silent loss raporu
eksik_ulkeler = analiz_verisi[analiz_verisi.isnull().any(axis=1)]
df_clean = analiz_verisi.dropna().copy()
df_clean.set_index('Location', inplace=True)

print("Silent Loss Raporu")
print(f"Veri Setindeki Toplam Ülke/Bölge Sayısı : {len(analiz_verisi)}")
print(f"Model Analizine Giren Net Ülke Sayısı  : {len(df_clean)}")
print(f"Eksik Veri Nedeniyle Çıkarılan Ülke Sayısı: {len(eksik_ulkeler)}")
if len(eksik_ulkeler) > 0:
    print(f"Çıkarılan Ülke(ler): {list(eksik_ulkeler['Location'].values)}")
print("="*60 + "\n")

# mutlak tfr'ı yüzdesel yapma
df_clean['TFR_Change_Pct'] = ((df_clean['TFR_2019'] - df_clean['TFR_1990']) / df_clean['TFR_1990']) * 100
df_clean['LE60_Change'] = df_clean['LE60_2019'] - df_clean['LE60_1990']
df_clean['Covid_Shock'] = df_clean['LE60_2019'] - df_clean['LE60_2021']

# Model Değişkenleri
features = ['TFR_2019', 'TFR_Change_Pct', 'LE60_2019', 'LE60_Change', 'Covid_Shock']
X = df_clean[features]

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# Korelasyon Matrisi
corr_matrix = df_clean[features].corr()
print("Değişkenler Arası İlişki Haritası")
print("="*60)
print(corr_matrix.round(3))
print("="*60 + "\n")

# K-Means Kümeleme (k=3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_std)

skor = silhouette_score(X_std, df_clean['Cluster'])
print(f" K-Means (k=3) Silhouette Skoru: {skor:.4f} \n")

# Kümeleri Sıralama ve İsimlendirme
kume_tfr = df_clean.groupby('Cluster')['TFR_2019'].mean().sort_values()
sirali_kodlar = kume_tfr.index.tolist()
isim_sozlugu = {
    sirali_kodlar[0]: 'Constrictive population',
    sirali_kodlar[1]: 'Stationary population',
    sirali_kodlar[2]: 'Expansive population'
}
df_clean['Clusters'] = df_clean['Cluster'].map(isim_sozlugu)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

df_clean['PCA_1'] = X_pca[:, 0]
df_clean['PCA_2'] = X_pca[:, 1]
var_ratio = pca.explained_variance_ratio_ * 100

loadings = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1_Loading', 'PC2_Loading'], 
    index=features
)

print("PCA Bileşen Yükleri ve Açıklanan Varyans")
print("="*60)
print(loadings.round(3))
print(f"\nAçıklanan Toplam Varyans: %{var_ratio[0]+var_ratio[1]:.1f}")
print("="*60 + "\n")

# Görselleştirme
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_clean, 
    x='PCA_1', y='PCA_2', 
    hue='Clusters', 
    palette='Set1', 
    s=120, alpha=0.7, edgecolor='w'
)

plt.title('Global Demographic Clusters and Core Representative Countries', fontsize=14, fontweight='bold')
plt.xlabel(f'Demographic Development Level (Life Expectancy vs. Fertility) | Explained Variance: %{var_ratio[0]:.1f}', fontsize=11)
plt.ylabel(f'Recent Shocks & Change (COVID Impact & Fertility Drop) | Explained Variance: %{var_ratio[1]:.1f}', fontsize=11)

# Centroidleri ve en yakın ülkeleri otomatik işaretleme
centroids = kmeans.cluster_centers_
dists = cdist(X_std, centroids)

# üst üste binmeyi önlemek için
offsets = {
    'Isle of Man': (0.3, 0.3),
    'United Kingdom': (0.3, -0.4),
    'Luxembourg': (-1.2, 0.4),
    'Türkiye': (0.3, 0.3),
    'Argentina': (0.3, -0.3),
    'Tunisia': (-0.9, -0.3),
    'Vanuatu': (0.3, 0.3),
    'Solomon Islands': (0.3, -0.3),
    'Senegal': (-0.8, -0.4)
}

for i, c_idx in enumerate(sirali_kodlar):
    cluster_name = isim_sozlugu[c_idx]
    df_clean[f'dist_{c_idx}'] = dists[:, c_idx]
    closest_countries = df_clean[df_clean['Cluster'] == c_idx].sort_values(f'dist_{c_idx}').head(3).index.tolist()
    
    for ulke in closest_countries:
        x_koord = df_clean.loc[ulke, 'PCA_1']
        y_koord = df_clean.loc[ulke, 'PCA_2']
        dx, dy = offsets.get(ulke, (0.3, 0.3))
        
        plt.annotate(
            ulke, 
            xy=(x_koord, y_koord), 
            xytext=(x_koord + dx, y_koord + dy),
            fontsize=9, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=1.2)
        )

plt.legend(title='Demographic Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()