import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Greenspace Impact Dashboard")

st.title("🌿 Greenspace Impact Monte Carlo Dashboard")

st.markdown("""
This dashboard identifies high-impact areas for **Good Karma Gardens (GKG)**. We use a **Monte Carlo simulation** to account for the fact that different grantors or stakeholders might weight environmental factors differently.
""")

# ----------------------------
# 1. Load & Clean Data
# ----------------------------

@st.cache_data
def load_and_clean_all_data():
    # Load the Original EJSM data as requested
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_Origonal.csv")
    
    # Load the other factors (DangerZone versions or originals)
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_DangerZone.csv")
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')

    # --- CLEANING & STANDARDIZING ---
    
    # Standardize EJSM (Original Data Processing)
    df_ejsm.columns = df_ejsm.columns.str.strip()
    # Ensure GEOID is a 11-digit string (California starts with '06')
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # Standardize other factor IDs
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)
    
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Function to create a 0-1 scale for the Monte Carlo
    def standardize_col(df, col):
        return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df_ejsm['score_std'] = standardize_col(df_ejsm, 'CIscore')
    df_income['score_std'] = standardize_col(df_income, 'med_hh_income')
    df_snap['score_std'] = standardize_col(df_snap, 'TractSNAP')
    df_heat['score_std'] = standardize_col(df_heat, 'DegHourDay')

    # Merge into a master simulation dataframe
    df_combined = df_ejsm[['GEOID10','score_std']] \
        .merge(df_income[['GEOID10','score_std']], on='GEOID10', how='outer', suffixes=('_ejsm', '_inc')) \
        .merge(df_snap[['GEOID10','score_std']], on='GEOID10', how='outer') \
        .merge(df_heat[['GEOID10','score_std']], on='GEOID10', how='outer', suffixes=('_snap', '_heat')) \
        .fillna(0)

    return df_ejsm, df_income, df_snap, df_heat, df_ziptract, df_combined

try:
    df_ejsm, df_income, df_snap, df_heat, df_ziptract, df_combined = load_and_clean_all_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----------------------------
# 2. Simulation Logic
# ----------------------------

zip_used = st.text_input("Enter ZIP Code:", "91505")
matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    st.error(f"ZIP code {zip_used} not found.")
    st.stop()

target_geoid = matching.iloc[0]['GEOID10']

# Monte Carlo setup
x = df_combined[['score_std_ejsm','score_std_inc','score_std_snap','score_std_heat']].to_numpy()
num_boot = 5000
weights = np.random.uniform(0, 1, (num_boot, 4))
weights /= weights.sum(axis=1, keepdims=True)
y = np.dot(weights, x.T) 

ymed = np.median(y, axis=0)
yp25 = np.percentile(y, 25, axis=0)
yp75 = np.percentile(y, 75, axis=0)

# ----------------------------
# 3. Main Dashboard Visuals
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
        dist = y[:, indices[0]]
        mu, std = norm.fit(dist)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(dist, bins=40, density=True, color='lightgrey', alpha=0.6, edgecolor='white')
        xmin, xmax = ax2.get_xlim()
        x_range = np.linspace(xmin, xmax, 100)
        ax2.plot(x_range, norm.pdf(x_range, mu, std), 'k', linewidth=2)
        ax2.axvline(mu, color='navy', label=f'Mean: {mu:.3f}')
        ax2.axvline(mu + std, color='navy', linestyle='--', label='+1 SD')
        ax2.axvline(mu - std, color='navy', linestyle='--', label='-1 SD')
        ax2.legend([f"Score of {zip_used}: {mu:.3f} ± {std:.3f}"], loc='upper right')
        st.pyplot(fig2)
        st.markdown(f"> ### **Key Finding**\n> Based on 5,000 simulations, the impact score for ZIP **{zip_used}** is **{mu:.3f} ± {std:.3f}**.")
    else:
        st.warning(f"Data for Tract {target_geoid} is incomplete.")

st.divider()

# ----------------------------
# 4. Component Deep-Dive: EJSM
# ----------------------------

st.header("🔍 Component Deep-Dive")
st.markdown("""
To understand the final score, we look at the individual factors. Below is the breakdown for the **Environmental Justice (EJSM)** score, 
calculated directly from the original LA County dataset.
""")

st.subheader("1. Environmental Justice Screening Method (EJSM)")

# Calculate EJSM stats from the original data
ejsm_raw = df_ejsm['CIscore'].dropna()
mean_val = ejsm_raw.mean()
std_val = ejsm_raw.std()
danger_threshold = mean_val - std_val
danger_zone_count = len(df_ejsm[df_ejsm['CIscore'] < danger_threshold])

col_left, col_right = st.columns([1, 2])

with col_left:
    st.info("**Methodology:**")
    st.write("""
    The EJSM score (CIscore) ranges from 4 to 20. It combines hazard proximity, health risk, 
    and social vulnerability. GKG identifies a **'Danger Zone'** as any tract scoring more than 1 Standard Deviation 
    below the mean, signifying extreme lack of green equity.
    """)
    
    st.table(pd.DataFrame({
        "Metric": ["Mean CIscore", "Standard Deviation", "Danger Threshold", "Tracts in Danger Zone"],
        "Value": [f"{mean_val:.2f}", f"{std_val:.2f}", f"{danger_threshold:.2f}", str(danger_zone_count)]
    }))

with col_right:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax3.hist(ejsm_raw, bins=20, color='#1f77b4', edgecolor='white', alpha=0.7)

    # Apply the "Danger Zone" coloring from your Spyder logic
    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < danger_threshold:
            patches[i].set_facecolor("red")

    # Overlay Normal Curve (scaled to frequency)
    x_pdf = np.linspace(min(ejsm_raw), max(ejsm_raw), 500)
    pdf = norm.pdf(x_pdf, mean_val, std_val)
    bin_width = bins[1] - bins[0]
    ax3.plot(x_pdf, pdf * len(ejsm_raw) * bin_width, color='orange', linewidth=2.5, label='Normal Distribution')

    # Annotation lines
    ax3.axvline(mean_val, color='black', linewidth=1.5)
    ax3.axvline(mean_val + std_val, color='black', linestyle=':')
    ax3.axvline(mean_val - std_val, color='black', linestyle=':')
    
    y_max = max(counts)
    ax3.text(mean_val, y_max*0.95, "Mean", rotation=90, ha='right')
    ax3.text(mean_val - std_val, y_max*0.95, "Danger Zone Start (-1 SD)", color='red', rotation=90, ha='right')

    ax3.set_title("Original EJSM Score Distribution", fontsize=14)
    ax3.set_xlabel("CIscore (4-20)")
    ax3.set_ylabel("Number of Census Tracts")
    st.pyplot(fig3)
