import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Greenspace Impact Dashboard")

st.title("🌿 Greenspace Impact Monte Carlo Dashboard")

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_DangerZone.csv")
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_DangerZone.csv")
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')

    # Standardize GEOIDs to 11-digit strings
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)
    
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    return df_ejsm, df_income, df_snap, df_heat, df_ziptract

try:
    df_ejsm, df_income, df_snap, df_heat, df_ziptract = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----------------------------
# User Input
# ----------------------------

zip_used = st.text_input("Enter ZIP Code:", "91505")
matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    st.error(f"ZIP code {zip_used} not found.")
    st.stop()

target_geoid = matching.iloc[0]['GEOID10']

# ----------------------------
# Processing & Monte Carlo
# ----------------------------

def standardize(df, col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df_ejsm['CIscore_std'] = standardize(df_ejsm, 'CIscore')
df_income['med_hh_income_std'] = standardize(df_income, 'med_hh_income')
df_snap['TractSNAP_std'] = standardize(df_snap, 'TractSNAP')
df_heat['DegHourDay_std'] = standardize(df_heat, 'DegHourDay')

df_combined = df_ejsm[['GEOID10','CIscore_std']] \
    .merge(df_income[['GEOID10','med_hh_income_std']], on='GEOID10', how='outer') \
    .merge(df_snap[['GEOID10','TractSNAP_std']], on='GEOID10', how='outer') \
    .merge(df_heat[['GEOID10','DegHourDay_std']], on='GEOID10', how='outer') \
    .fillna(0)

x = df_combined[['CIscore_std','med_hh_income_std','TractSNAP_std','DegHourDay_std']].to_numpy()

num_boot = 5000
weights = np.random.uniform(0, 1, (num_boot, 4))
weights /= weights.sum(axis=1, keepdims=True)

# y contains 5000 impact scores for every tract
y = np.dot(weights, x.T) 

# Calculate stats across the region
ymed = np.median(y, axis=0)
yp25 = np.percentile(y, 25, axis=0)
yp75 = np.percentile(y, 75, axis=0)

# ----------------------------
# Plot 1: Regional Impact Ranking
# ----------------------------

st.subheader("📊 Regional Impact Ranking & Uncertainty")

# Sort everything based on the median score
sort_idx = np.argsort(ymed)
sorted_med = ymed[sort_idx]
sorted_p25 = yp25[sort_idx]
sorted_p75 = yp75[sort_idx]

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sorted_med, color='#1b9e77', label='Median Impact Score', linewidth=2)
ax1.fill_between(range(len(sorted_med)), sorted_p25, sorted_p75, 
                 color='lightskyblue', alpha=0.4, label='Uncertainty (25th-75th Percentile)')

ax1.set_title("Impact Scores across LA Census Tracts (Sorted Ranking)", fontsize=12)
ax1.set_ylabel("Impact Score")
ax1.set_xlabel("Ranked Census Tracts")
ax1.legend(loc='upper left')
ax1.grid(alpha=0.2)
st.pyplot(fig1)

# ----------------------------
# Plot 2: ZIP Code Specifics
# ----------------------------

st.subheader(f"🎯 Impact Score Statistics for ZIP {zip_used}")

indices = np.where(df_combined['GEOID10'].values == target_geoid)[0]

if len(indices) > 0:
    geo_distribution = y[:, indices[0]]
    mu, std = norm.fit(geo_distribution)
    
    # Legend Text Formatting
    legend_label = f"Score of {zip_used}: {mu:.3f} ± {std:.3f}"

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # 1. Histogram
    count, bins, ignored = ax2.hist(geo_distribution, bins=40, density=True, 
                                     color='lightgrey', alpha=0.6, edgecolor='white')
    
    # 2. Fitted Normal Curve
    xmin, xmax = ax2.get_xlim()
    x_range = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x_range, mu, std)
    ax2.plot(x_range, p, 'k', linewidth=2)

    # 3. Statistical Lines
    ax2.axvline(mu, color='navy', linestyle='-', linewidth=2)
    ax2.axvline(mu + std, color='navy', linestyle='--', linewidth=1.5)
    ax2.axvline(mu - std, color='navy', linestyle='--', linewidth=1.5)

    # Legend & Formatting
    ax2.set_title(f"Simulation Density for {zip_used} (Tract: {target_geoid})")
    ax2.set_xlabel("Weighted Impact Score")
    ax2.set_ylabel("Probability Density")
    
    # Custom Legend as requested
    ax2.legend([legend_label], loc='upper right', frameon=True, fontsize=12)
    
    st.pyplot(fig2)

    # Heavy Emphasis Note
    st.markdown(f"""
    > ### **Key Finding** > Based on 5,000 Monte Carlo simulations, the impact score for ZIP code **{zip_used}** is:  
    > ## **{mu:.3f} ± {std:.3f}**
    """)
else:
    st.warning(f"Data for Tract {target_geoid} (ZIP {zip_used}) is incomplete in the environment datasets.")
