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
    # 1. Load EJSM (Original)
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_Original.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Load Income (Original)
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_original.csv")
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Load Heat (Original)
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat = df_heat.dropna(subset=['DegHourDay'])
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # 4. Load SNAP (Original)
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_Original.csv")
    df_snap.columns = df_snap.columns.str.strip()
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    # Calculate percentage and handle cases with 0 population
    df_snap = df_snap[(df_snap['Pop2010'] > 0) & (df_snap['TractSNAP'].notna())]
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)

    # 5. Load Crosswalk
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization helper (0-1 scale)
    def standardize_col(df, col, invert=False):
        scaled = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return 1 - scaled if invert else scaled

    df_ejsm['score_std'] = standardize_col(df_ejsm, 'CIscore')
    df_income['score_std'] = standardize_col(df_income, 'med_hh_income', invert=True) # Low income = High impact
    df_heat['score_std'] = standardize_col(df_heat, 'DegHourDay') # High heat = High impact
    df_snap['score_std'] = standardize_col(df_snap, 'SNAP_pct') # High SNAP % = High impact

    # Merge into master simulation dataframe
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

# Monte Carlo (5k runs)
x = df_combined[['score_std_ejsm','score_std_inc','score_std_snap','score_std_heat']].to_numpy()
num_boot = 5000
weights = np.random.uniform(0, 1, (num_boot, 4))
weights /= weights.sum(axis=1, keepdims=True)
y = np.dot(weights, x.T) 

ymed = np.median(y, axis=0)
yp25 = np.percentile(y, 25, axis=0)
yp75 = np.percentile(y, 75, axis=0)

# ----------------------------
# 3. Dashboard Visuals
# ----------------------------

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Regional Impact Ranking & Uncertainty")
    sort_idx = np.argsort(ymed)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ymed[sort_idx], color='#1b9e77', linewidth=2)
    ax1.fill_between(range(len(ymed)), yp25[sort_idx], yp75[sort_idx], color='lightskyblue', alpha=0.4)
    ax1.set_ylabel("Weighted Impact Score")
    ax1.set_xlabel("Ranked Census Tracts")
    st.pyplot(fig1)

with col2:
    st.subheader(f"🎯 Impact Statistics for ZIP {zip_used}")
    indices = np.where(df_combined['GEOID10'].values == target_geoid)[0]
    if len(indices) > 0:
        dist = y[:, indices[0]]
        mu, std = norm.fit(dist)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(dist, bins=40, density=True, color='lightgrey', alpha=0.6, edgecolor='white')
        xmin, xmax = ax2.get_xlim(); xr = np.linspace(xmin, xmax, 100)
        ax2.plot(xr, norm.pdf(xr, mu, std), 'k', linewidth=2)
        ax2.axvline(mu, color='navy', lw=2); ax2.axvline(mu+std, color='navy', ls='--'); ax2.axvline(mu-std, color='navy', ls='--')
        ax2.legend([f"Score: {mu:.3f} ± {std:.3f}"], loc='upper right')
        st.pyplot(fig2)
        st.markdown(f"> ### **Key Finding**\n> Based on simulation, the impact score for ZIP **{zip_used}** is **{mu:.3f} ± {std:.3f}**.")
    else:
        st.warning("Data incomplete for this tract.")

st.divider()

# ----------------------------
# 4. Component Deep-Dive
# ----------------------------

st.header("🔍 Component Deep-Dive")

# Row 1: EJSM & Income
c_ejsm, c_inc = st.columns(2)
with c_ejsm:
    st.subheader("1. Environmental Justice (EJSM)")
    raw = df_ejsm['CIscore'].dropna()
    m, s = raw.mean(), raw.std()
    t = m - s
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(raw, bins=20, color='#1f77b4', alpha=0.7)
    for p in ax.patches: 
        if p.get_x() < t: p.set_facecolor("red")
    ax.axvline(m, color='black'); ax.axvline(t, color='red', ls='--')
    st.pyplot(fig)
    st.caption(f"Mean: {m:.2f} | Danger Threshold (<-1 SD): {t:.2f}")

with c_inc:
    st.subheader("2. Median Household Income")
    raw = df_income['med_hh_income'].dropna()
    m, s = raw.mean(), raw.std()
    t = m - s
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(raw, bins=500, color='#2ca02c', alpha=0.7)
    for p in ax.patches: 
        if p.get_x() < t: p.set_facecolor("red")
    ax.axvline(m, color='black'); ax.axvline(t, color='red', ls='--'); ax.set_xlim(0, 250000)
    st.pyplot(fig)
    st.caption(f"Mean: ${m:,.0f} | Danger Threshold (<-1 SD): ${t:,.0f}")

# Row 2: Heat & SNAP
c_heat, c_snap = st.columns(2)
with c_heat:
    st.subheader("3. Heat Burden (Degree Hour Days)")
    raw = df_heat['DegHourDay'].dropna()
    m, s = raw.mean(), raw.std()
    t = m + s
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(raw, bins=300, color='#d62728', alpha=0.7)
    for p in ax.patches: 
        if p.get_x() > t: p.set_facecolor("red")
    ax.axvline(m, color='black'); ax.axvline(t, color='red', ls='--')
    st.pyplot(fig)
    st.caption(f"Mean: {m:.2f} | Danger Threshold (>+1 SD): {t:.2f}")

with c_snap:
    st.subheader("4. Food Access (SNAP Participation %)")
    raw = df_snap['SNAP_pct'].dropna()
    m, s = raw.mean(), raw.std()
    t = m + s # HIGH SNAP is the danger zone
    
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax.hist(raw, bins=300, color='#9467bd', edgecolor='none', alpha=0.7)
    
    # Recolor danger zone (ABOVE threshold)
    for i in range(len(patches)):
        if (bins[i] + bins[i+1])/2 > t:
            patches[i].set_facecolor("red")
            
    # Add Normal Curve
    x_range = np.linspace(min(raw), max(raw), 500)
    ax.plot(x_range, norm.pdf(x_range, m, s) * len(raw) * (bins[1]-bins[0]), color='orange', lw=2)
    ax.axvline(m, color='black'); ax.axvline(t, color='red', ls='--')
    
    ax.set_xlabel("% of Population on SNAP")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    danger_count = len(df_snap[df_snap['SNAP_pct'] > t])
    st.caption(f"Mean: {m:.2f}% | Danger Threshold (>+1 SD): {t:.2f}% | Tracts in Danger: {danger_count}")

st.markdown("""
---
**Note for Grantors:** Each red 'Danger Zone' identifies communities where the lack of green space is 
compounded by significant environmental or economic stress. Good Karma Gardens targets these specific 
red-bin tracts to maximize community impact.
""")

