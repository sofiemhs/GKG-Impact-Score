import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")

# ----------------------------
# 1. Data Loading & Processing
# ----------------------------

@st.cache_data
def load_all_data():
    # Attempt to find the correct data directory path
    # This handles different folder structures on GitHub vs Streamlit Cloud
    possible_paths = ["data", "GKG-Impact-Score/data"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    
    if data_path is None:
        raise FileNotFoundError("Could not find the 'data' folder in the repository.")

    # EJSM (Original)
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Original.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # Income (Original)
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # Heat Burden (Original - Path Fixed)
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat = df_heat.dropna(subset=['DegHourDay'])
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # SNAP (Original)
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts (1).csv")
    df_snap.columns = df_snap.columns.str.strip()
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0) & (df_snap['TractSNAP'] != 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)

    # Crosswalk
    df_ziptract = pd.read_excel(f"{data_path}/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardize helper (0-1 scale) for Monte Carlo
    def std_col(df, col, inv=False):
        s = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return 1 - s if inv else s

    df_ejsm['s'] = std_col(df_ejsm, 'CIscore')
    df_income['s'] = std_col(df_income, 'med_hh_income', inv=True)
    df_heat['s'] = std_col(df_heat, 'DegHourDay')
    df_snap['s'] = std_col(df_snap, 'SNAP_pct')

    df_comb = df_ejsm[['GEOID10','s']].merge(df_income[['GEOID10','s']], on='GEOID10', how='outer', suffixes=('_e','_i')) \
              .merge(df_heat[['GEOID10','s']], on='GEOID10', how='outer') \
              .merge(df_snap[['GEOID10','s']], on='GEOID10', how='outer', suffixes=('_h','_s')).fillna(0)

    return df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb

# ----------------------------
# 2. Analytics Execution
# ----------------------------

try:
    df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb = load_all_data()
except Exception as e:
    st.error(f"❌ Error Loading Data: {e}")
    st.stop()

st.sidebar.header("User Input")
zip_in = st.sidebar.text_input("Enter target ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if not match.empty:
    target_geoid = match.iloc[0]['GEOID10']
    x = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    weights = np.random.uniform(0, 1, (5000, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    y = np.dot(weights, x.T)
    
    mu_all = np.median(y, axis=0)
    idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Regional Impact Ranking")
        fig, ax = plt.subplots()
        ax.plot(np.sort(mu_all), color='#1b9e77', label='LA County Tracts')
        ax.set_facecolor('#f8f9fa')
        ax.set_ylabel("Impact Score")
        ax.set_xlabel("Ranked Tracts")
        ax.legend()
        st.pyplot(fig)
        
    with col_b:
        st.subheader(f"Impact Statistics: ZIP {zip_in}")
        if len(idx) > 0:
            d = y[:, idx[0]]
            m, s = norm.fit(d)
            fig, ax = plt.subplots()
            ax.hist(d, bins=30, color='lightgrey', edgecolor='white', density=True)
            ax.axvline(m, color='red', lw=2, label=f'Mean Score: {m:.3f}')
            ax.set_xlabel("Calculated Impact Score")
            ax.legend()
            st.pyplot(fig)
            st.success(f"The simulated impact score for {zip_in} is **{m:.3f} ± {s:.3f}**.")
else:
    st.sidebar.warning("ZIP code not found in LA County database.")

st.divider()

# ----------------------------
# 3. Component Deep-Dives
# ----------------------------

st.header("🔍 Factor Distribution & Danger Zones")
st.markdown("Red bars indicate the **Danger Zone** (Tracts > 1 SD away from the mean).")

def plot_component(df, col, name, unit, is_high_danger=True, bins=100):
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    
    danger_zone = df[df[col] > thresh] if is_high_danger else df[df[col] < thresh]
    zone_sym = ">" if is_high_danger else "<"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(f"This factor measures {name.lower()} across LA County.")
        st.table(pd.DataFrame({
            "Metric": ["LA County Mean", "Standard Deviation", "Danger Threshold", "Tracts in Danger Zone"],
            "Details": [f"{mean_v:,.2f} {unit}", f"{std_v:,.2f}", f"{zone_sym} {thresh:,.2f}", str(len(danger_zone))]
        }))

    with c2:
        fig, ax = plt.subplots(figsize=(10, 4))
        counts, bins, patches = ax.hist(data, bins=bins, color='#bdc3c7', edgecolor='none', alpha=0.8)
        
        # Color Danger Zone Red
        for i in range(len(patches)):
            mid = (bins[i] + bins[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        # Bell Curve
        x_axis = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_axis, norm.pdf(x_axis, mean_v, std_v) * len(data) * (bins[1]-bins[0]), color='black', lw=1.5, label='Normal Distribution')
        
        ax.axvline(mean_v, color='black', ls='-', label='Mean')
        ax.axvline(thresh, color='#e74c3c', ls='--', lw=2, label=f'Danger Threshold ({zone_sym} 1SD)')
        
        ax.set_title(f"Distribution: {name}")
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

# Component Section Rendering
plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', is_high_danger=False, bins=20)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', '$', is_high_danger=False, bins=250)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'DegHrDays', is_high_danger=True, bins=150)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% Pop', is_high_danger=True, bins=150)

