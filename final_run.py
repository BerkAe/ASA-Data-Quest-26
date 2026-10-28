"""
FINAL RUN - single script that produces every number used in the slides / Q&A.
Run once, take all numbers from the printed report and from final_run_outputs/*.csv.

Usage (Windows):  set DATA_DIR=C:\\Work\\Asa Data quest 26   &  python final_run.py
Needs in DATA_DIR:
  WPP2024_Demographic_Indicators_Medium_csv.gz
  WPP2024_Supplement_Births_and_LE60.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, silhouette_score,
                             silhouette_samples, calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"C:\Work\Asa Data quest 26"
OUT_DIR = os.path.join(DATA_DIR, "final_run_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

K, SEED, N_SEEDS = 3, 42, 20
SMALL_POP_THOUSANDS = 100          # "very small unit" threshold (thousands, 2019)
FEATURES = ['TFR_2019', 'TFR_Change_Pct', 'LE60_2019', 'LE60_Change', 'Covid_Shock']


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def save(df, name, index=True):
    df.to_csv(os.path.join(OUT_DIR, name), index=index)


# ---------------------------------------------------------------- 1. load + merge (on LocID, not on name)
wpp = pd.read_csv(os.path.join(DATA_DIR, "WPP2024_Demographic_Indicators_Medium_csv.gz"), low_memory=False)
le = pd.read_csv(os.path.join(DATA_DIR, "WPP2024_Supplement_Births_and_LE60.csv"), low_memory=False)
wpp = wpp[wpp['LocTypeName'] == 'Country/Area']


def pick(df, year, col, new):
    return df.loc[df['Time'] == year, ['LocID', col]].rename(columns={col: new}).drop_duplicates('LocID')


base = wpp[['LocID', 'ISO3_code', 'Location']].drop_duplicates('LocID')
d = (base.merge(pick(wpp, 1990, 'TFR', 'TFR_1990'), on='LocID', how='left')
         .merge(pick(wpp, 2019, 'TFR', 'TFR_2019'), on='LocID', how='left')
         .merge(pick(le, 1990, 'LE60', 'LE60_1990'), on='LocID', how='left')
         .merge(pick(le, 2019, 'LE60', 'LE60_2019'), on='LocID', how='left')
         .merge(pick(le, 2021, 'LE60', 'LE60_2021'), on='LocID', how='left'))
# optional context columns (NOT used for clustering)
for col, new in [('TPopulation1July', 'Pop_2019_thousands'), ('MedianAgePop', 'MedianAge_2019')]:
    if col in wpp.columns:
        d = d.merge(pick(wpp, 2019, col, new), on='LocID', how='left')

# name consistency check between the two files
nm = le[['LocID', 'Location']].drop_duplicates('LocID').rename(columns={'Location': 'Loc_le'})
chk = base.merge(nm, on='LocID', how='outer')
mism = chk[(chk['Location'] != chk['Loc_le'])]
section("SILENT LOSS REPORT")
print(f"Country/Area units in WPP file          : {len(base)}")
print(f"Units only in one of the two files (LocID): {chk['Location'].isna().sum() + chk['Loc_le'].isna().sum()}")
print(f"LocIDs whose NAME differs between files   : {len(mism)}  {mism[['LocID','Location','Loc_le']].values.tolist()}")
core = ['TFR_1990', 'TFR_2019', 'LE60_1990', 'LE60_2019', 'LE60_2021']
dropped = d[d[core].isna().any(axis=1)]
print(f"Dropped for missing data                : {len(dropped)}  {dropped['Location'].tolist()}")
d = d.dropna(subset=core).copy().set_index('Location')
N = len(d)
print(f"N entering the analysis                 : {N}")
save(dropped, "dropped_units.csv", index=False)

# ---------------------------------------------------------------- 2. variables
d['TFR_Change_Pct'] = (d['TFR_2019'] - d['TFR_1990']) / d['TFR_1990'] * 100
d['TFR_Change_Abs'] = d['TFR_2019'] - d['TFR_1990']
d['LE60_Change'] = d['LE60_2019'] - d['LE60_1990']
d['Covid_Shock'] = d['LE60_2019'] - d['LE60_2021']

X = StandardScaler().fit_transform(d[FEATURES])
names = d.index.to_numpy()

section("CORRELATIONS")
print(d[FEATURES].corr().round(3))
save(d[FEATURES].corr().round(3), "correlations.csv")

# ---------------------------------------------------------------- 3. choice of k
section("CHOICE OF k (KMeans, seed 42, n_init=10)")
rows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    rows.append({'k': k, 'silhouette': silhouette_score(X, km.labels_),
                 'calinski_harabasz': calinski_harabasz_score(X, km.labels_),
                 'davies_bouldin': davies_bouldin_score(X, km.labels_), 'inertia': km.inertia_})
ksel = pd.DataFrame(rows).round(4)
print(ksel.to_string(index=False))
save(ksel, "k_selection.csv", index=False)

# ---------------------------------------------------------------- 4. reference model (k=3)
km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X)
order = np.argsort(d.groupby(km.labels_)['TFR_2019'].mean().to_numpy())     # 0 = lowest mean TFR
remap = {old: new for new, old in enumerate(order)}
ref = np.array([remap[l] for l in km.labels_])
centers = km.cluster_centers_[order]
d['Cluster'] = ref        # 0 = lowest mean TFR_2019, 2 = highest. Naming is a separate, documented decision.

section("REFERENCE CLUSTERING (0 = lowest mean TFR_2019 ... 2 = highest)")
sil = silhouette_samples(X, ref)
d['Silhouette_i'] = sil
print(f"Overall silhouette: {sil.mean():.4f}")
print("Cluster sizes:", d['Cluster'].value_counts().sort_index().to_dict())
print("Mean silhouette per cluster:", pd.Series(sil).groupby(ref).mean().round(3).to_dict())
print(f"Units with NEGATIVE silhouette (arguably misassigned): {(sil < 0).sum()}")
print("Lowest-silhouette units:", d['Silhouette_i'].nsmallest(8).round(3).to_dict())

prof_cols = FEATURES + [c for c in ['TFR_Change_Abs', 'Pop_2019_thousands', 'MedianAge_2019'] if c in d.columns]
profile = d.groupby('Cluster')[prof_cols].mean().round(2)
profile.insert(0, 'N', d['Cluster'].value_counts().sort_index())
zprof = pd.DataFrame(X, columns=FEATURES, index=d.index).groupby(d['Cluster']).mean().round(2)
print("\nCluster means (original units):\n", profile.to_string())
print("\nCluster means (standardized z) -> use for the slide-4 chart:\n", zprof.to_string())
if 'MedianAge_2019' in d.columns:
    print("\nMedian age (NOT used in clustering) per cluster -> external check of cluster names:")
    print(d.groupby('Cluster')['MedianAge_2019'].describe().round(1).to_string())
save(profile, "cluster_profile_original_units.csv")
save(zprof, "cluster_profile_zscores.csv")

# ---------------------------------------------------------------- 5. PCA
section("PCA")
pca = PCA(n_components=2).fit(X)
d[['PC1', 'PC2']] = pca.transform(X)
load = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1', 'PC2']).round(3)
print(load.to_string())
print("Explained variance %:", (pca.explained_variance_ratio_ * 100).round(1).tolist(),
      "| total:", round(pca.explained_variance_ratio_.sum() * 100, 1))
save(load, "pca_loadings.csv")

# ---------------------------------------------------------------- 6. outliers
section("MOST EXTREME UNITS (largest |z| on any variable)")
Z = pd.DataFrame(X, columns=FEATURES, index=d.index)
ext = Z.abs().max(axis=1).nlargest(6).index
print(Z.loc[ext].round(2).to_string())
print("PC1/PC2 of the extreme PC point:", d.loc[d['PC1'].abs().idxmax(), ['PC1', 'PC2']].round(2).to_dict(),
      "->", d['PC1'].abs().idxmax())

# ---------------------------------------------------------------- 7. representatives + seed stability
def match(a, b):
    cm = confusion_matrix(a, b, labels=range(K))
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum(), dict(zip(c, r)), cm


def reps(Xm, labels, ctr, n=3):
    out = {}
    for c in range(K):
        idx = np.where(labels == c)[0]
        out[c] = list(names[idx[np.argsort(np.linalg.norm(Xm[idx] - ctr[c], axis=1))[:n]]])
    return out


ref_reps = reps(X, ref, centers)
section("REPRESENTATIVE COUNTRIES (3 nearest to centroid in 5-D) + SEED STABILITY")
for c, r in ref_reps.items():
    print(f"Cluster {c}: {r}")
same = {c: 0 for c in range(K)}
partition_same, max_changed = 0, 0
for s in range(N_SEEDS):
    m = KMeans(n_clusters=K, random_state=s, n_init=10).fit(X)
    lab, ctr = m.labels_, m.cluster_centers_
    kept, mp, _ = match(ref, lab)
    lab_m = np.array([mp[l] for l in lab])
    ctr_m = np.zeros_like(ctr)
    for o, n_ in mp.items():
        ctr_m[n_] = ctr[o]
    rr = reps(X, lab_m, ctr_m)
    for c in range(K):
        same[c] += set(rr[c]) == set(ref_reps[c])
    max_changed = max(max_changed, N - kept)
print(f"Seeds (of {N_SEEDS}) giving the identical representative set: {same}")
print(f"Max units changing cluster across {N_SEEDS} seeds: {max_changed}")

# ---------------------------------------------------------------- 8. ARI table
def compare(other):
    kept, mp, cm = match(ref, other)
    return adjusted_rand_score(ref, other), N - kept, cm


section("SENSITIVITY TABLE (reference = KMeans, all 5 features, seed 42)")
X_nocovid = StandardScaler().fit_transform(d[[f for f in FEATURES if f != 'Covid_Shock']])
feat_abs = ['TFR_2019', 'TFR_Change_Abs', 'LE60_2019', 'LE60_Change', 'Covid_Shock']
X_abs = StandardScaler().fit_transform(d[feat_abs])
lab_nc = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_nocovid)
models = {
    'KMeans, Covid_Shock removed': lab_nc,
    'KMeans, ABSOLUTE TFR change (old definition)': KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_abs),
    'GMM': GaussianMixture(n_components=K, random_state=SEED, n_init=5).fit_predict(X),
    'Ward': AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(X),
}
tab = []
for nme, lab in models.items():
    ari, ch, _ = compare(lab)
    tab.append([nme, round(ari, 3), ch, round((N - ch) / N * 100, 1)])
noise = [compare(KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(X))[:2] for s in range(N_SEEDS)]
tab.append([f'KMeans, other seeds (min ARI / max changed over {N_SEEDS})',
            round(min(n[0] for n in noise), 3), max(n[1] for n in noise),
            round((N - max(n[1] for n in noise)) / N * 100, 1)])
# very small units excluded
keep = (d['Pop_2019_thousands'] >= SMALL_POP_THOUSANDS).to_numpy() if 'Pop_2019_thousands' in d.columns else None
if keep is not None:
    Xs = StandardScaler().fit_transform(d.loc[keep, FEATURES])
    ls = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(Xs)
    kept, _, _ = match(ref[keep], ls)
    tab.append([f'KMeans, units < {SMALL_POP_THOUSANDS}k pop removed (n={keep.sum()}; compared on common units)',
                round(adjusted_rand_score(ref[keep], ls), 3), int(keep.sum() - kept),
                round(kept / keep.sum() * 100, 1)])
t = pd.DataFrame(tab, columns=['Comparison', 'ARI', 'Units changing cluster', '% staying'])
print(t.to_string(index=False))
save(t, "ari_table.csv", index=False)

_, mp_nc, cm = match(ref, lab_nc)
col_order = [c for r in range(K) for c, rr in mp_nc.items() if rr == r]      # columns aligned to reference labels
cm = cm[:, col_order]
trans = pd.DataFrame(cm, index=[f'ref {i}' for i in range(K)], columns=[f'no-Covid {i}' for i in range(K)])
print("\nWho moves when Covid_Shock is removed? (rows = reference cluster, columns matched by Hungarian algorithm;"
      " off-diagonal = units that change cluster)")
print(trans.to_string())
save(trans, "transitions_no_covid.csv")

# ---------------------------------------------------------------- 9. small / non-sovereign units
if 'Pop_2019_thousands' in d.columns:
    section("SMALL UNITS")
    small = d[d['Pop_2019_thousands'] < SMALL_POP_THOUSANDS]
    print(f"Units with population < {SMALL_POP_THOUSANDS}k in 2019: {len(small)}")
    print(small.groupby('Cluster').size().to_dict(), "(by cluster)")
    print(small.index.tolist())
print("\nNOTE: sovereignty is NOT in these files. Flag sovereign states with an external list "
      "(e.g. UN member states) using final_run_outputs/country_level.csv (has ISO3_code).")

# ---------------------------------------------------------------- 10. save country-level table
save(d.reset_index(), "country_level.csv", index=False)
print(f"\nAll tables saved to: {OUT_DIR}")
