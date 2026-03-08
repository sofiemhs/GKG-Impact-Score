import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Greenspace Impact Dashboard")

st.title("🌿 Greenspace Impact Monte Carlo Dashboard")

st.markdown("""
This dashboard identifies high-impact areas for **Good Karma Gardens (GKG)** builds. By analyzing where environmental and 
socioeconomic factors overlap, GKG can strategically prioritize projects in tracts that benefit most from green space[cite: 231, 234].
""")

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    # Primary Data Files
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_DangerZone.csv")
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_DangerZone.csv")
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')

    # Standardize GEOIDs to 11-digit strings for consistent merging
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
# User Input & Identification
# ----------------------------

zip_used = st.text_input("Enter ZIP Code:", "91505")
matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    st.error(f"ZIP code {zip_used} not found.")
    st.stop()

target_geoid = matching.iloc[0]['GEOID10']

# ----------------------------
# Monte Carlo Processing
# ----------------------------

def standardize(df, col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Prepare standardized columns for the Monte Carlo simulation [cite: 357]
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

# Run Monte Carlo (5,000 simulations) to account for weighting uncertainty [cite: 359, 361]
num_boot = 5000
weights = np.random.uniform(0, 1, (num_boot, 4))
weights /= weights.sum(axis=1, keepdims=True)
y = np.dot(weights, x.T) 

ymed = np.median(y, axis=0)
yp25 = np.percentile(y, 25, axis=0)
yp75 = np.percentile(y, 75, axis=0)

# ----------------------------
# MAIN DASHBOARD OUTPUT
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Regional Impact Ranking & Uncertainty")
    sort_idx = np.argsort(ymed)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ymed[sort_idx], color='#1b9e77', label='Median Impact Score', linewidth=2)
    ax1.fill_between(range(len(ymed)), yp25[sort_idx], yp75[sort_idx], color='lightskyblue', alpha=0.4, label='Uncertainty (25th-75th Percentile)')
    ax1.set_ylabel("Impact Score")
    ax1.set_xlabel("Ranked Census Tracts")
    ax1.legend(loc='upper left')
    st.pyplot(fig1)

with col2:
    st.subheader(f"🎯 Impact Statistics for ZIP {zip_used}")
    indices = np.where(df_combined['GEOID10'].values == target_geoid)[0]
    if len(indices) > 0:
        geo_distribution = y[:, indices[0]]
        mu, std = norm.fit(geo_distribution)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(geo_distribution, bins=40, density=True, color='lightgrey', alpha=0.6, edgecolor='white')
        xmin, xmax = ax2.get_xlim()
        x_range = np.linspace(xmin, xmax, 100)
        ax2.plot(x_range, norm.pdf(x_range, mu, std), 'k', linewidth=2)
        ax2.axvline(mu, color='navy', linestyle='-', linewidth=2)
        ax2.axvline(mu + std, color='navy', linestyle='--', linewidth=1.5)
        ax2.axvline(mu - std, color='navy', linestyle='--', linewidth=1.5)
        ax2.legend([f"Score of {zip_used}: {mu:.3f} ± {std:.3f}"], loc='upper right')
        st.pyplot(fig2)
        st.markdown(f"> ### **Key Finding**\n> Based on simulation, the impact score for ZIP **{zip_used}** is **{mu:.3f} ± {std:.3f}**.")
    else:
        st.warning(f"Data for Tract {target_geoid} is incomplete in the environment datasets.")

st.divider()

# ----------------------------
# NEW SECTION: DANGER ZONE EXPLANATIONS
# ----------------------------

st.header("🔍 Understanding the Components")
st.markdown("""
The final impact score is composed of four main factors. To identify "Danger Zones," we analyze the distribution of each 
original dataset and flag areas that deviate significantly from the mean[cite: 232, 233].
""")

# --- 1. Environmental Justice (EJSM) ---
st.subheader("1. Environmental Justice Screening Method (EJSM)")

col_ejsm_left, col_ejsm_right = st.columns([1, 2])

with col_ejsm_left:
    st.markdown("""
    **Metric:** CIscore  
    Developed by USC and Occidental College, the **EJSM** evaluates environmental justice across four categories[cite: 246]:
    * Hazard Proximity and Sensitive Land Use
    * Health Risk and Exposure
    * Social and Health Vulnerability
    * Climate Change Vulnerability (Tree Canopy)
    
    Tracts are scored from 4 to 20. We define the **Danger Zone** as tracts with scores more than one standard deviation below 
    the mean, identifying areas that would benefit most from additional green space[cite: 255, 257, 258].
    """)
    
    # Calculate EJSM Stats for Display
    ejsm_data = df_ejsm['CIscore'].dropna()
    ejsm_mean = ejsm_data.mean()
    ejsm_std = ejsm_data.std()
    ejsm_threshold = ejsm_mean - ejsm_std
    ejsm_danger_count = len(df_ejsm[df_ejsm['CIscore'] < ejsm_threshold])

    # Display Summary Table
    st.table(pd.DataFrame({
        "Metric": ["Mean", "Standard Deviation", "Danger Threshold (<-1 SD)", "Tracts in Danger Zone"],
        "Value": [f"{ejsm_mean:.2f}", f"{ejsm_std:.2f}", f"{ejsm_threshold:.2f}", str(ejsm_danger_count)]
    }))

with col_ejsm_right:
    # Recreate the EJSM Histogram using user's Spyder logic
    fig_ejsm, ax_ejsm = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax_ejsm.hist(ejsm_data, bins=20, density=False, color='#1f77b4', edgecolor='white')

    # Recolor bins in the Danger Zone
    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < ejsm_threshold:
            patches[i].set_facecolor("red")

    # Add Normal Curve
    x_curve = np.linspace(min(ejsm_data), max(ejsm_data), 500)
    pdf = norm.pdf(x_curve, ejsm_mean, ejsm_std)
    bin_width = bins[1] - bins[0]
    ax_ejsm.plot(x_curve, pdf * len(ejsm_data) * bin_width, color='orange', linewidth=2)

    # Add Mean and SD lines
    ax_ejsm.axvline(ejsm_mean, color='black', label='Mean')
    ax_ejsm.axvline(ejsm_mean + ejsm_std, color='black', linestyle='--', label='+1 SD')
    ax_ejsm.axvline(ejsm_mean - ejsm_std, color='black', linestyle='--', label='-1 SD')
    
    # Text labels
    y_max = max(counts)
    ax_ejsm.text(ejsm_mean, y_max*0.9, "Mean", rotation=90, va='top', ha='right')
    ax_ejsm.text(ejsm_mean + ejsm_std, y_max*0.9, "+1 SD", rotation=90, va='top', ha='right')
    ax_ejsm.text(ejsm_mean - ejsm_std, y_max*0.9, "-1 SD", rotation=90, va='top', ha='right')

    ax_ejsm.set_xlabel("EJSM CIscore")
    ax_ejsm.set_ylabel("Frequency (Count of Census Tracts)")
    ax_ejsm.set_title("Distribution of EJSM Scores with Highlighted Danger Zone")
    st.pyplot(fig_ejsm)
