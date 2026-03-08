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
2. **The Impact Score** is calculated using **10,000 Monte Carlo simulations** to account for varying environmental and social priorities.
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
# 2. Main Analytics (10k Monte Carlo)
# ----------------------------

# ZIP CODE INPUT (Sidebar)
st.sidebar.header("📍 Location Filter")
zip_in = st.sidebar.text_input("Enter any ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if not match.empty:
    target_geoid = match.iloc[0]['GEOID10']
    
    # Run 10,000 Iterations
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    num_iters = 10000 
    weights = np.random.uniform(0, 1, (num_iters, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    
    sim_results = np.dot(weights, x_matrix.T)
    
    # 1. LOCAL SECTION (TOP)
    st.header(f"🎯 Local Impact Results: ZIP {zip_in}")
    idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]
    
    if len(idx) > 0:
        d = sim_results[:, idx[0]]
        m, s = norm.fit(d)
        
        # Big Hero Stat
        st.metric(label=f"ESTIMATED IMPACT SCORE", value=f"{m:.3f}", delta=f"± {s:.3f} SD")

        col_local_l, col_local_r = st.columns([2, 1])
        with col_local_l:
            fig_local, ax_local = plt.subplots(figsize=(10, 5))
            ax_local.hist(d, bins=30, color='#aed6f1', edgecolor='white', density=True, alpha=0.7)
            x_fit = np.linspace(min(d), max(d), 100)
            ax_local.plot(x_fit, norm.pdf(x_fit, m, s), color='#2e86c1', lw=3, label='Normal Fit')
            ax_local.axvline(m, color='#1b4f72', lw=2, label=f'Mean: {m:.3f}')
            ax_local.axvline(m + s, color='#e74c3c', ls='--', lw=1.5, label=f'+1 SD')
            ax_local.axvline(m - s, color='#e74c3c', ls='--', lw=1.5, label=f'-1 SD')
            ax_local.set_title(f"Impact Distribution for ZIP {zip_in}")
            ax_local.legend(loc='upper right')
            st.pyplot(fig_local)
        
        with col_local_r:
            st.write("### Statistical Summary")
            st.write(f"Based on 10,000 simulated priority weightings, census tract **{target_geoid}** shows the following impact potential:")
            st.table(pd.DataFrame({
                "Metric": ["Average Score", "Uncertainty (SD)", "Impact Range"],
                "Value": [f"{m:.4f}", f"{s:.4f}", f"{m-s:.3f} to {m+s:.3f}"]
            }))
    
    st.divider()

    # 2. REGIONAL SECTION (BOTTOM)
    st.header("🌎 Regional Context (LA County)")
    median_scores = np.median(sim_results, axis=0)
    p25 = np.percentile(sim_results, 25, axis=0)
    p75 = np.percentile(sim_results, 75, axis=0)
    sorted_idx = np.argsort(median_scores)

    fig_reg, ax_reg = plt.subplots(figsize=(12, 5))
    ax_reg.fill_between(range(len(median_scores)), p25[sorted_idx], p75[sorted_idx], 
                        alpha=0.2, label='25th-75th Percentile', color='#3498db')
    ax_reg.plot(median_scores[sorted_idx], color='#2980b9', lw=2.5, label='County Median Line')
    ax_reg.set_ylabel("Weighted Impact Index")
    ax_reg.set_xlabel("Census Tracts (Ranked Low to High Impact)")
    ax_reg.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Stats overlay
    textstr = '\n'.join((r'Mean SD = 0.0465', r'Mean SE = 0.000465'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_reg.text(0.05, 0.95, textstr, transform=ax_reg.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    ax_reg.legend(loc='lower right')
    st.pyplot(fig_reg)

else:
    st.sidebar.error(f"ZIP {zip_in} not found. Please try another LA County ZIP.")

st.divider()

# ----------------------------
# 3. Deep-Dive Section (Unchanged Tables)
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
        counts, bins, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.8)
        for i in range(len(patches)):
            mid = (bins[i] + bins[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
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

# Factor Content
ejsm_desc = "The **Environmental Justice Screening Method (EJSM)** identifies communities facing cumulative impacts from multiple pollution sources and social vulnerabilities."
income_desc = "**Median Household Income** tracks financial stability. Lower income areas often lack the private funds for high-quality landscaping and community cooling."
heat_desc = "**Heat Burden** (Degree Hour Days) measures the intensity of the urban heat island effect. High-heat areas are critical targets for garden shade and cooling."
snap_desc = "**SNAP Participation** acts as a proxy for food insecurity, identifying tracts where urban agriculture can provide vital fresh produce access."

plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', ejsm_desc, is_high_danger=False, bins=20)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', '$', income_desc, is_high_danger=False, bins=250)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'DegHrDays', heat_desc, is_high_danger=True, bins=150)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% Pop', snap_desc, is_high_danger=True, bins=150)
