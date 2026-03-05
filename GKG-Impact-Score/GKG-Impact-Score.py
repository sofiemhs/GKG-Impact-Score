import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------
# Upload data
# ----------------------------

zip_used = input("Enter ZIP code of where Greenspace is located: ")
print("ZIP entered:", zip_used)

EJSM = "GKG-Impact-Score/data/EJSM_DangerZone.csv"
Income = "GKG-Impact-Score/data/Income_DangerZone.csv"
SNAP = r"C:/Users/sofie/OneDrive/Desktop/TractSNAP_DangerZone.csv"
Heat = "GKG-Impact-Score/data/DegHourDays_DangerZone.csv"
ZIP_TRACT = r"C:/Users/sofie/OneDrive/Desktop/ZIP_TRACT_122025.xlsx"

df_ejsm = pd.read_csv(EJSM)
df_income = pd.read_csv(Income)
df_snap = pd.read_csv(SNAP)
df_heat = pd.read_csv(Heat)
df_ziptract = pd.read_excel(ZIP_TRACT)

# ----------------------------
# Clean ZIP → TRACT file
# ----------------------------

df_ziptract = df_ziptract.rename(columns={'ZIP': 'ZIP', 'TRACT': 'GEOID10'})
df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
df_ziptract['GEOID10'] = df_ziptract['GEOID10'].astype(str)

# Convert ZIP to GEOID
matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    print("ZIP code not found in mapping file.")
    exit()

# If ZIP maps to multiple tracts, take first one
target_geoid = matching.iloc[0]['GEOID10']
print("Mapped GEOID:", target_geoid)

# ----------------------------
# Standardize GEOID column name
# ----------------------------

df_ejsm = df_ejsm.rename(columns={'Tract_1': 'GEOID10'})
df_income = df_income.rename(columns={'tract': 'GEOID10'})
df_snap = df_snap.rename(columns={'CT10': 'GEOID10'})
df_heat = df_heat.rename(columns={'FIPS': 'GEOID10'})

df_ejsm['GEOID10'] = df_ejsm['GEOID10'].astype(str)
df_income['GEOID10'] = df_income['GEOID10'].astype(str)
df_heat['GEOID10'] = df_heat['GEOID10'].astype(str)

# SNAP fix
df_snap['GEOID10'] = df_snap['GEOID10'].astype(str).str.replace('.0', '', regex=False)
df_snap['GEOID10'] = '6037' + df_snap['GEOID10']

# ----------------------------
# Standardize columns (0–1 scaling)
# ----------------------------

def standardize_column(df, col):
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val - min_val != 0:
        return (df[col] - min_val) / (max_val - min_val)
    else:
        return pd.Series([0]*len(df))

df_ejsm['CIscore_std'] = standardize_column(df_ejsm, 'CIscore')
df_income['med_hh_income_std'] = standardize_column(df_income, 'med_hh_income')
df_snap['TractSNAP_std'] = standardize_column(df_snap, 'TractSNAP')
df_heat['DegHourDay_std'] = standardize_column(df_heat, 'DegHourDay')

# ----------------------------
# Merge standardized columns
# ----------------------------

df_ejsm_std = df_ejsm[['GEOID10', 'CIscore_std']]
df_income_std = df_income[['GEOID10', 'med_hh_income_std']]
df_snap_std = df_snap[['GEOID10', 'TractSNAP_std']]
df_heat_std = df_heat[['GEOID10', 'DegHourDay_std']]

df_combined = df_ejsm_std.merge(df_income_std, on='GEOID10', how='outer') \
                         .merge(df_snap_std, on='GEOID10', how='outer') \
                         .merge(df_heat_std, on='GEOID10', how='outer')

df_combined = df_combined.fillna(0)

# ----------------------------
# Convert to numpy arrays
# ----------------------------

x1 = df_combined['CIscore_std'].to_numpy()
x2 = df_combined['med_hh_income_std'].to_numpy()
x3 = df_combined['TractSNAP_std'].to_numpy()
x4 = df_combined['DegHourDay_std'].to_numpy()

# ----------------------------
# Monte Carlo Simulation
# ----------------------------

num_boot = 10000
y = np.zeros((num_boot, len(x1)))

coefficients = np.random.uniform(0, 1, (num_boot, 4))
coefficients = coefficients / coefficients.sum(axis=1, keepdims=True)

for n in range(num_boot):
    y[n, :] = (coefficients[n, 0]*x1 +
               coefficients[n, 1]*x2 +
               coefficients[n, 2]*x3 +
               coefficients[n, 3]*x4)

# ----------------------------
# Uncertainty Statistics
# ----------------------------

ymed = np.median(y, axis=0)
y25 = np.percentile(y, 25, axis=0)
y75 = np.percentile(y, 75, axis=0)
ystd = np.std(y, axis=0)
y_se = ystd / np.sqrt(num_boot)

mean_sd = np.mean(ystd)
mean_se = np.mean(y_se)

# ----------------------------
# Plot 1
# ----------------------------

plt.close('all')
plt.figure(figsize=(9,6))

sorted_indices = np.argsort(ymed)
ymed_sorted = ymed[sorted_indices]
y25_sorted = y25[sorted_indices]
y75_sorted = y75[sorted_indices]

plt.fill_between(range(len(ymed_sorted)), y25_sorted, y75_sorted,
                 alpha=0.2, label='25th–75th Percentile')

plt.plot(ymed_sorted, linewidth=2, label='Median')

plt.xlabel('Census Tracts (Sorted)')
plt.ylabel('Weighted Standardized Index')
plt.title('Monte Carlo Simulation of Weighted Index')

plt.legend()

plt.text(0.02, 0.95,
         f"Mean SD = {mean_sd:.4f}\nMean SE = {mean_se:.6f}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", alpha=0.3))

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# Distribution for Selected ZIP (mapped GEOID)
# ----------------------------

indices = np.where(df_combined['GEOID10'].to_numpy() == target_geoid)[0]

if len(indices) == 0:
    print("Mapped GEOID not found in combined dataset.")
    exit()

original_index = indices[0]

geo_distribution = y[:, original_index]

mu = np.mean(geo_distribution)
sigma = np.std(geo_distribution)

x_vals = np.linspace(min(geo_distribution), max(geo_distribution), 500)
normal_curve = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals-mu)/sigma)**2)

plt.figure(figsize=(9,6))

plt.hist(geo_distribution, bins=50, density=True, alpha=0.6,
         label='Monte Carlo Distribution')
plt.plot(x_vals, normal_curve, linewidth=2,
         label='Normal Approximation')

plt.axvline(mu, linewidth=3, label='Mean')
plt.axvline(mu + sigma, linestyle='--', linewidth=2, label='+1 SD')
plt.axvline(mu - sigma, linestyle='--', linewidth=2, label='-1 SD')

plt.xlabel('Impact Score')
plt.ylabel('Density')
plt.title(f'Monte Carlo Uncertainty Distribution\nZIP {zip_used}')

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.figtext(0.5, 0.01,
            f"Impact Score for ZIP {zip_used} = {mu:.3f} ± {sigma:.3f}",
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle="round", alpha=0.3))


plt.show()


