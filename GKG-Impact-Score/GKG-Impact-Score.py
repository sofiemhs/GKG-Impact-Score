import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- HEADER SECTION ---
st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")

st.markdown("""
### **How to use this tool:**
1. **Enter a ZIP Code** in the sidebar to see the specific impact score for that neighborhood.
2. **The Impact Score** is calculated using 5,000 Monte Carlo simulations to account for varying environmental and social priorities.
3. **Explore the Deep-Dives** below to see how each individual factor (Heat, Income, etc.) is distributed across LA County. 
4. **Red Bins** in the graphs represent the **'Danger Zone'**—tracts that are more than 1 Standard Deviation away from the County mean.
""")

# ----------------------------
# 1. Data Loading & Processing
# ----------------------------

@st.cache_data
def load_all_data():
    possible_paths = ["data", "GKG-Impact-Score/data"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    
    if data_path is None:
        raise FileNotFoundError("Could not find the 'data' folder.")

    # EJSM
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # Income
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # Heat Burden
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat = df_heat.dropna(subset=['DegHourDay'])
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # SNAP
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

try:
    df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb = load_all_data()
except Exception as e:
    st.error(f"❌ Error Loading Data: {e}")
    st.stop()

# ----------------------------
# 2. Main Analytics
# ----------------------------

st.sidebar.header("📍 Location Filter")
zip_in = st.sidebar.text_input("Enter ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if not match.empty:
    target_geoid = match.iloc[0]['GEOID10']
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    weights = np.random.uniform(0, 1, (5000, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T)
    
    mu_all = np.median(sim_results, axis=0)
    idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Regional Impact Ranking")
        fig, ax = plt.subplots()
        ax.plot(np.sort(mu_all), color='#1b9e77', label='LA County Tracts')
        ax.set_ylabel("Impact Score")
        ax.legend()
        st.pyplot(fig)
        
    with col_b:
        st.subheader(f"Local Impact Stats: {zip_in}")
        if len(idx) > 0:
            d = sim_results[:, idx[0]]
            m, s = norm.fit(d)
            fig, ax = plt.subplots()
            ax.hist(d, bins=30, color='lightgrey', edgecolor='white', density=True)
            ax.axvline(m, color='red', lw=2, label=f'Mean Impact: {m:.3f}')
            ax.legend()
            st.pyplot(fig)
            st.success(f"**ZIP {zip_in} Analysis:** Average Impact Score is {m:.3f} with a margin of {s:.3f}.")

st.divider()

# ----------------------------
# 3. Deep-Dive Section
# ----------------------------

st.header("🔍 Factor Descriptions & Distributions")

def plot_component(df, col, name, unit, description, is_high_danger=True, bins=100):
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    
    danger_zone = df[df[col] > thresh] if is_high_danger else df[df[col] < thresh]
    zone_sym = ">" if is_high_danger else "<"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(description)
        st.table(pd.DataFrame({
            "Metric": ["LA County Mean", "Standard Deviation", "Danger Threshold", "Tracts in Danger Zone"],
            "Details": [f"{mean_v:,.2f} {unit}", f"{std_v:,.2f}", f"{zone_sym} {thresh:,.2f}", str(len(danger_zone))]
        }))

    with c2:
        fig, ax = plt.subplots(figsize=(10, 4))
        counts, bins, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.8, density=False)
        
        # Color Red for Danger
        for i in range(len(patches)):
            mid = (bins[i] + bins[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        # Properly Scaled Bell Curve
        x = np.linspace(data.min(), data.max(), 500)
        pdf = norm.pdf(x, mean_v, std_v)
        bin_width = bins[1] - bins[0]
        ax.plot(x, pdf * len(data) * bin_width, color='black', lw=2, label='Normal Curve')
        
        ax.axvline(mean_v, color='black', ls='-', label='Mean')
        ax.axvline(thresh, color='#e74c3c', ls='--', lw=2, label='Danger Threshold')
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

# --- Define Factor Descriptions ---
ejsm_desc = "The **Environmental Justice Screening Method** identifies communities facing cumulative impacts from multiple pollution sources and social vulnerabilities."
income_desc = "**Median Household Income** tracks financial stability. Lower income areas often lack the private funds for high-quality landscaping and community cooling."
heat_desc = "**Heat Burden** (Degree Hour Days) measures the intensity of the urban heat island effect. High-heat areas are critical targets for garden shade and cooling."
snap_desc = "**SNAP Participation** acts as a proxy for food insecurity. Tracts with high SNAP enrollment benefit most from urban agriculture and fresh produce access."

# --- Render Sections ---
plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', ejsm_desc, is_high_danger=False, bins=20)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', '$', income_desc, is_high_danger=False, bins=250)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'DegHrDays', heat_desc, is_high_danger=True, bins=150)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% Pop', snap_desc, is_high_danger=True, bins=150)
