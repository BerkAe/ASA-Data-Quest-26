import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# kendine göre değiş
DATA_DIR = os.environ.get("DATA_DIR", r"C:\Work\Asa Data quest 26")
OUT_DIR = os.environ.get("OUT_DIR", DATA_DIR)
os.chdir(DATA_DIR)

K = 3
SEED = 42
N_NOISE_SEEDS = 20

df_wpp = pd.read_csv("WPP2024_Demographic_Indicators_Medium_csv.gz", low_memory=False)
df_le60 = pd.read_csv("WPP2024_Supplement_Births_and_LE60.csv", low_memory=False)

df_wpp = df_wpp[df_wpp['LocTypeName'] == 'Country/Area']


def al(df, yil, kolon, yeni):
    return df[df['Time'] == yil][['Location', kolon]].rename(columns={kolon: yeni})


tum_ulkeler = pd.DataFrame({'Location': df_wpp['Location'].unique()})
df = (tum_ulkeler
      .merge(al(df_wpp, 1990, 'TFR', 'TFR_1990'), on='Location', how='left')
      .merge(al(df_wpp, 2019, 'TFR', 'TFR_2019'), on='Location', how='left')
      .merge(al(df_le60, 1990, 'LE60', 'LE60_1990'), on='Location', how='left')
      .merge(al(df_le60, 2019, 'LE60', 'LE60_2019'), on='Location', how='left')
      .merge(al(df_le60, 2021, 'LE60', 'LE60_2021'), on='Location', how='left'))

df_clean = df.dropna().copy().set_index('Location')

df_clean['TFR_Change_Pct'] = (df_clean['TFR_2019'] - df_clean['TFR_1990']) / df_clean['TFR_1990'] * 100
df_clean['LE60_Change'] = df_clean['LE60_2019'] - df_clean['LE60_1990']
df_clean['Covid_Shock'] = df_clean['LE60_2019'] - df_clean['LE60_2021']

features_all = ['TFR_2019', 'TFR_Change_Pct', 'LE60_2019', 'LE60_Change', 'Covid_Shock']
features_no_covid = ['TFR_2019', 'TFR_Change_Pct', 'LE60_2019', 'LE60_Change']

X_all = StandardScaler().fit_transform(df_clean[features_all])
X_no_covid = StandardScaler().fit_transform(df_clean[features_no_covid])
N = len(df_clean)


# fonksiyon tanımlama
def eslesen_ulke_sayisi(ref, diger):
    """Etiket numaraları keyfi olduğu için, kümeleri Macar algoritmasıyla
    en iyi şekilde eşleştirip aynı kümede kalan ülke sayısını döndürür."""
    cm = confusion_matrix(ref, diger)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum()


def karsilastir(ref, diger):
    ari = adjusted_rand_score(ref, diger)
    kalan = eslesen_ulke_sayisi(ref, diger)
    return ari, kalan / N * 100, N - kalan


#Modeller
ref = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_all)

modeller = {
    'K-Means (Covid_Shock removed)':
        KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_no_covid),
    'GMM (Gaussian Mixture)':
        GaussianMixture(n_components=K, random_state=SEED, n_init=5).fit_predict(X_all),
    'Ward (Hierarchical Clustering)':
        AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X_all),
}

# Gürültü tabanı: aynı model, sadece farklı rastgele başlangıç
noise = [karsilastir(ref, KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(X_all))
         for s in range(N_NOISE_SEEDS)]
noise_ari_min = min(n[0] for n in noise)
noise_degisen_max = max(n[2] for n in noise)

satirlar = []
for ad, etiket in modeller.items():
    ari, uyum, degisen = karsilastir(ref, etiket)
    satirlar.append([ad, f"{ari:.3f}", f"{uyum:.1f}%", f"{degisen}"])

satirlar.append(['K-Means, different seed (noise)',
                 f">= {noise_ari_min:.3f}", f"{(N - noise_degisen_max) / N * 100:.1f}%+", f"<= {noise_degisen_max}"])

# Covid etkisi: ilk satır
degisen_covid = int(satirlar[0][3])

print(f"Countries in analysis: {N}")
for s in satirlar:
    print(s)

#Görselleştirme
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis('off')

basliklar = ['Comparison Model\n(Reference: K-Means, all features)',
             'ARI Score', 'Countries Staying\nin Same Cluster', 'Countries Changing\nCluster']
# "Aynı Kümede Kalan" sütunu: ülke yüzdesi, değiştiren: sayı
tablo = ax.table(cellText=satirlar, colLabels=basliklar, loc='center',
                 cellLoc='center', colWidths=[0.4, 0.15, 0.2, 0.25])
tablo.auto_set_font_size(False)
tablo.set_fontsize(11)
tablo.scale(1, 2.4)

for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    elif col == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#ECF0F1')
    if row == len(satirlar):  # gürültü tabanı satırı
        cell.set_text_props(style='italic')

ax.set_title('Methodological Sensitivity and Cluster Overlap (ARI)',
             fontsize=14, fontweight='bold', pad=20)

alt_yazi = (f"When Covid_Shock is removed from the model, {degisen_covid} of {N} countries "
            f"change cluster.")

fig.subplots_adjust(top=0.85, bottom=0.2)
fig.text(0.5, 0.06, alt_yazi, ha='center', va='center', fontsize=11,
         fontweight='bold', color='#c0392b',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9ebea', edgecolor='#e6b0aa'))

cikis = os.path.join(OUT_DIR, "Slayt2_Tablo_ve_Covid.png")
fig.savefig(cikis, dpi=300, bbox_inches='tight')
print("Saved:", cikis)
plt.show()