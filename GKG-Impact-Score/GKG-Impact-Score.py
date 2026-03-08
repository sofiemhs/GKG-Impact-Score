import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Greenspace Impact Dashboard")

st.title("🌿 Greenspace Impact Monte Carlo Dashboard")

st.markdown("""
This dashboard estimates the **impact score of a census tract**
based on environmental justice, income, SNAP usage, and heat burden.
A Monte Carlo simulation randomly weights these factors to account
for uncertainty in how they might be prioritized.
""")

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    # Use the paths you provided
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_DangerZone.csv")
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_DangerZone.csv")
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')

    # FIX: Standardize GEOID/Tract columns to 11-character strings immediately
    # California FIPS starts with 06, so we pad to 11 digits
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)
    
    # Clean Zip-Tract Crosswalk
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    return df_ejsm, df_income, df_snap, df_heat, df_ziptract

# Load datasets
try:
    df_ejsm, df_income, df_snap, df_heat, df_ziptract = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# ----------------------------
# User Input
# ----------------------------

zip_used = st.text_input("Enter ZIP Code:", "91505")

# Find the tract corresponding to the ZIP
matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    st.error(f"ZIP code {zip_used} not found in the reference directory.")
    st.stop()

# Use the most likely tract for that ZIP
target_geoid = matching.iloc[0]['GEOID10']

# ----------------------------
# Standardize columns
# ----------------------------

def standardize(df, col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df_ejsm['CIscore_std'] = standardize(df_ejsm, 'CIscore')
df_income['med_hh_income_std'] = standardize(df_income, 'med_hh_income')
df_snap['TractSNAP_std'] = standardize(df_snap, 'TractSNAP')
df_heat['DegHourDay_std'] = standardize(df_heat, 'DegHourDay')

# Merge all environmental data
df_combined = df_ejsm[['GEOID10','CIscore_std']] \
    .merge(df_income[['GEOID10','med_hh_income_std']], on='GEOID10', how='outer') \
    .merge(df_snap[['GEOID10','TractSNAP_std']], on='GEOID10', how='outer') \
    .merge(df_heat[['GEOID10','DegHourDay_std']], on='GEOID10', how='outer') \
    .fillna(0)

# Prepare numpy array for matrix multiplication
x = df_combined[['CIscore_std','med_hh_income_std','TractSNAP_std','DegHourDay_std']].to_numpy()

# ----------------------------
# Monte Carlo Simulation
# ----------------------------

num_boot = 5000
# Generate random weights for the 4 factors
coefficients = np.random.uniform(0, 1, (num_boot, 4))
# Normalize weights so they sum to 1
coefficients /= coefficients.sum(axis=1, keepdims=True)

# Calculate weighted scores for all tracts
y = np.dot(coefficients, x.T)
ymed = np.median(y, axis=0)

# ----------------------------
# Visualizations
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Regional Impact Ranking")
    sorted_scores = np.sort(ymed)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sorted_scores, color='#1b9e77', linewidth=2)
    ax1.set_title("Median Impact Scores across LA Census Tracts", fontsize=12)
    ax1.set_ylabel("Impact Score (Normalized)")
    ax1.set_xlabel("Census Tracts (Ranked Low to High)")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.subheader(f"🎯 ZIP Code {zip_used} Distribution")
    
    # Locate the tract in our combined data
    indices = np.where(df_combined['GEOID10'].values == target_geoid)[0]

    if len(indices) > 0:
        # Get all 5000 simulated scores for this specific tract
        geo_distribution = y[:, indices[0]]
        
        mu = np.mean(geo_distribution)
        p25 = np.percentile(geo_distribution, 25)
        p75 = np.percentile(geo_distribution, 75)

        st.write(f"**Tract ID:** {target_geoid}")
        
        # Plotting the Distribution
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        # Histogram
        n, bins, patches = ax2.hist(geo_distribution, bins=40, density=True, 
                                     color='lightgrey', edgecolor='white', alpha=0.7)
        
        # Solid Mean Line
        ax2.axvline(mu, color='navy', linestyle='-', linewidth=2.5, label=f'Mean: {mu:.3f}')
        
        # Shaded area between 25th and 75th percentile
        ax2.axvspan(p25, p75, color='lightskyblue', alpha=0.4, label='Interquartile Range (25th-75th)')
        
        # Labeling
        ax2.set_title(f"Impact Score Uncertainty for ZIP {zip_used}", fontsize=12)
        ax2.set_xlabel("Weighted Impact Score")
        ax2.set_ylabel("Probability Density")
        ax2.legend(loc='upper right')
        
        st.pyplot(fig2)
        
        # Stats summary
        st.info(f"The mean impact score is **{mu:.3f}**. 50% of the simulated weightings fall between **{p25:.3f}** and **{p75:.3f}**.")
    else:
        st.warning(f"ZIP {zip_used} maps to Tract {target_geoid}, but this tract is missing from the environmental data files.")
