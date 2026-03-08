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
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Load Income (Original)
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_original.csv")
    df_income.columns = df_income.columns.str.strip()
    # Cleaning as per your logic: remove commas/percents and handle $0 values
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Load Remaining Factors
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx", engine='openpyxl')

    # Standardize other factor IDs
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization helper (0-1 scale)
    def standardize_col(df, col, invert=False):
        scaled = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        # If invert=True, lower values (like low income) get HIGHER impact scores
        return 1 - scaled if invert else scaled

    df_ejsm['score_std'] = standardize_col(df_ejsm, 'CIscore')
    df_income['score_std'] = standardize_col(df_income, 'med_hh_income', invert=True)
    df_snap['score_std'] = standardize_col(df_snap, 'TractSNAP')
    df_heat['score_std'] = standardize_col(df_heat, 'DegHourDay')

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

# Monte Carlo setup (5k runs)
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
# 4. Component Deep-Dive
# ----------------------------

st.header("🔍 Component Deep-Dive")
st.markdown("We isolate each factor to show how 'Danger Zones' were calculated from the raw datasets.")

# --- EJSM SECTION ---
st.subheader("1. Environmental Justice Screening Method (EJSM)")
ejsm_raw = df_ejsm['CIscore'].dropna()
m1, s1 = ejsm_raw.mean(), ejsm_raw.std()
t1 = m1 - s1
c1_left, c1_right = st.columns([1, 2])

with c1_left:
    st.write("Identifies hazard proximity and social vulnerability. Red bins represent high-priority tracts.")
    st.table(pd.DataFrame({"Metric": ["Mean", "SD", "Threshold"], "Value": [f"{m1:.2f}", f"{s1:.2f}", f"{t1:.2f}"]}))

with c1_right:
    fig_e, ax_e = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax_e.hist(ejsm_raw, bins=20, color='#1f77b4', edgecolor='white', alpha=0.7)
    for i in range(len(patches)):
        if (bins[i] + bins[i+1])/2 < t1: patches[i].set_facecolor("red")
    x_e = np.linspace(min(ejsm_raw), max(ejsm_raw), 500)
    ax_e.plot(x_e, norm.pdf(x_e, m1, s1) * len(ejsm_raw) * (bins[1]-bins[0]), color='orange')
    ax_e.axvline(m1, color='black'); ax_e.axvline(m1-s1, color='red', linestyle='--')
    st.pyplot(fig_e)

# --- INCOME SECTION ---
st.subheader("2. Median Household Income")
inc_raw = df_income['med_hh_income'].dropna()
m2, s2 = inc_raw.mean(), inc_raw.std()
t2 = m2 - s2 # Low income is the danger zone

c2_left, c2_right = st.columns([1, 2])

with c2_left:
    st.info("**Methodology:**")
    st.write("""
    Income is a primary indicator of 'Green Gentrification' and resource access. 
    Tracts earning less than one standard deviation below the LA mean are flagged as **Danger Zones**.
    """)
    st.table(pd.DataFrame({
        "Metric": ["Mean Income", "Standard Deviation", "Danger Threshold", "Tracts in Danger Zone"],
        "Value": [f"${m2:,.0f}", f"${s2:,.0f}", f"${t2:,.0f}", str(len(df_income[df_income['med_hh_income'] < t2]))]
    }))

with c2_right:
    # Plotting histogram with 500 bins as requested
    fig_i, ax_i = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax_i.hist(inc_raw, bins=500, color='#2ca02c', edgecolor='none', alpha=0.7)

    # Recolor danger zone
    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < t2:
            patches[i].set_facecolor("red")

    # Add Normal Curve
    x_i = np.linspace(min(inc_raw), max(inc_raw), 500)
    pdf_i = norm.pdf(x_i, m2, s2)
    ax_i.plot(x_i, pdf_i * len(inc_raw) * (bins[1]-bins[0]), color='orange', linewidth=2)

    # Annotations
    ax_i.axvline(m2, color='black', label='Mean')
    ax_i.axvline(t2, color='red', linestyle='--', label='Danger Zone Start')
    
    ax_i.set_title("Distribution of Median Household Income", fontsize=14)
    ax_i.set_xlabel("Income ($)")
    ax_i.set_ylabel("Frequency")
    ax_i.set_xlim(0, 250000) # Keep scale readable
    st.pyplot(fig_i)
