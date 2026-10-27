import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Değişkenler:
# LE60 = Life Expectancy at age 60
# TFR = Total Fertility Rate (doğurganlık oranı)
# TFR_2019 --> 2019 Toplam Doğurganlık Oranı (Kadın başına ortalama çocuk sayısı)
# TFR_Change --> 1990-2019 arası TFR değişimi (TFR_2019 - TFR_1990)
# LE60_2019 --> 2019'da 60 yaşındaki kişinin kalan ortalama ömrü (Yıl)
# LE60_Change --> 1990-2019 arası LE60 değişimi (LE60_2019 - LE60_1990)
# Covid_Shock --> COVID-19 kaynaklı LE60 kaybı (LE60_2019 - LE60_2021)

# dosyaların dizini (değiştirin kendize göre)
os.chdir(r"C:\Work")

# veri okuma (dosya adlarını kendinize göre değiştirin)
df_wpp = pd.read_csv("WPP2024_Demographic_Indicators_Medium_csv.gz", low_memory=False)
df_le60 = pd.read_csv("WPP2024_Supplement_Births_and_LE60.csv", low_memory=False)

# filtre
df_wpp = df_wpp[df_wpp['LocTypeName'] == 'Country/Area']

# Verileri alma ve birleştirme
tfr_1990 = df_wpp[df_wpp['Time'] == 1990][['Location', 'TFR']].rename(columns={'TFR': 'TFR_1990'})
tfr_2019 = df_wpp[df_wpp['Time'] == 2019][['Location', 'TFR']].rename(columns={'TFR': 'TFR_2019'})

le_1990 = df_le60[df_le60['Time'] == 1990][['Location', 'LE60']].rename(columns={'LE60': 'LE60_1990'})
le_2019 = df_le60[df_le60['Time'] == 2019][['Location', 'LE60']].rename(columns={'LE60': 'LE60_2019'})
le_2021 = df_le60[df_le60['Time'] == 2021][['Location', 'LE60']].rename(columns={'LE60': 'LE60_2021'})

analiz_verisi = tfr_1990.merge(tfr_2019, on='Location', how='inner')
for df in [le_1990, le_2019, le_2021]:
    analiz_verisi = analiz_verisi.merge(df, on='Location', how='inner')

# Yeni değişkenleri hesaplama
analiz_verisi['TFR_Change'] = analiz_verisi['TFR_2019'] - analiz_verisi['TFR_1990']
analiz_verisi['LE60_Change'] = analiz_verisi['LE60_2019'] - analiz_verisi['LE60_1990']
analiz_verisi['Covid_Shock'] = analiz_verisi['LE60_2019'] - analiz_verisi['LE60_2021']


# temizlik ve filtre
df_clean = analiz_verisi.dropna().copy()
df_clean.set_index('Location', inplace=True)

features = ['TFR_2019', 'TFR_Change', 'LE60_2019', 'LE60_Change', 'Covid_Shock']
X = df_clean[features]


# scaler
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)


# k=3 için K-Means kümeleme
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_std)

# Silhouette skor
skor = silhouette_score(X_std, df_clean['Cluster'])
print(f"--- Silhouette Score for K=3: {skor:.4f} ---\n")

# Kümeleri anlamlandırma
kume_tfr = df_clean.groupby('Cluster')['TFR_2019'].mean().sort_values()
sirali_kodlar = kume_tfr.index.tolist()

isim_sozlugu = {
    sirali_kodlar[0]: 'Constrictive population',
    sirali_kodlar[1]: 'Stationary population',
    sirali_kodlar[2]: 'Expansive population'
}
df_clean['Clusters'] = df_clean['Cluster'].map(isim_sozlugu)


# PCA ile görselleştirme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

df_clean['PCA_1'] = X_pca[:, 0]
df_clean['PCA_2'] = X_pca[:, 1]
var_ratio = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_clean, 
    x='PCA_1', y='PCA_2', 
    hue='Clusters', 
    palette='Set1', 
    s=120, alpha=0.7, edgecolor='w'
)

plt.title('Demographic Distribution of Country Clusters (K=3) and Core Examples', fontsize=14)
plt.xlabel(f'Current Demographic Status (Life Expectancy and Fertility) - Variance explained: %{var_ratio[0]:.1f}', fontsize=11)
plt.ylabel(f'Historical Change Rate and Covid Shock - Variance explained: %{var_ratio[1]:.1f}', fontsize=11)

# örnek ülkeleri  yazdırdığımız yer
istisnai_ulke = df_clean['PCA_1'].idxmin()
ornek_ulkeler = {
    'Constrictive population': ['Japan', 'Germany', 'Italy'], # veya gelişmiş
    'Stationary population': ['Türkiye', 'Turkey', 'Brazil', 'India'], # veya gelişmekte olan
    'Expansive population': ['Nigeria', 'Somalia', 'Afghanistan', istisnai_ulke] # veya gelişmemiş ülkeler
}

# Aşağıdaki algoritma örnek ülkeleri etiketleyip grafiği yerleştiriyor.
# Ve ülke data sette var mı diye checkliyor.
try:
    for grup, ulkeler in ornek_ulkeler.items():
        if grup in df_clean['Clusters'].unique(): 
            for ulke in ulkeler:
                if ulke in df_clean.index: 
                    x_koordinati = df_clean.loc[ulke, 'PCA_1']
                    y_koordinati = df_clean.loc[ulke, 'PCA_2']

                    # Ok çıkararak noktayı gösterme (annotate ile arrowprops)
                    plt.annotate(
                        ulke, 
                        xy=(x_koordinati, y_koordinati), # Okun ucu (Noktanın tam üstü)
                        xytext=(x_koordinati + 0.3, y_koordinati + 0.3), # Yazının duracağı yer
                        fontsize=9,
                        fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=1.2)
                    )
except Exception as e:
    print(f"Error, there was an error during naming most likely cause is mismatch {e}")


plt.legend(title='Demographic Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# Kümeler
ozet_tablo = df_clean.groupby('Clusters')[features].mean().round(2)
print("Variable mean for each cluster: \n")
print(ozet_tablo)