import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- HEADER SECTION ---
st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")

with st.expander("📖 About This Project & Methodology (Click to Expand)"):
    st.markdown("""
    ### **Our Research Question**
    *What impact are GKG builds having, and where is the need for green space conversion greatest?* [cite: 4]
    
    ### **The Methodology**
    To answer this, we developed a **'Danger Zone' Index** using key factors that influence the necessity of community gardens: SNAP participation, EJSM Social Vulnerability, Heat Days, and Income[cite: 6]. 
    
    We utilize a **Monte Carlo Simulation** (10,000 iterations) to calculate an overall impact score for every census tract in Los Angeles County[cite: 135, 136]. By shifting the 'weights' of importance between Environmental, Social, and Financial factors 10,000 times, we ensure our high-impact areas remain high-impact regardless of which specific metric is prioritized[cite: 132, 133].
    
    ### **How to Use:**
    1. **Enter a ZIP Code** in the sidebar to see the specific impact score for that neighborhood.
    2. **View the Local Variability:** See how much the impact score "swings" based on different priorities.
    3. **Regional Context:** See where your ZIP stands compared to the rest of the 2,000+ tracts in LA County.
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

    # EJSM (Environmental Justice Screening Method)
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # Income Data
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # Heat Burden (Safe Clean Water Program)
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat = df_heat.dropna(subset=['DegHourDay'])
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # SNAP (Food Access)
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts (1).csv")
    df_snap.columns = df_snap.columns.str.strip()
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0) & (df_snap['TractSNAP'] != 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)

    # ZIP to Tract Crosswalk
    df_ziptract = pd.read_excel(f"{data_path}/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization (0 to 1 Scale)
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
zip_in = st.sidebar.text_input("Enter any ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if not match.empty:
    target_geoid = match.iloc[0]['GEOID10']
    
    # 10,000 Iteration Monte Carlo
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    num_iters = 10000 
    weights = np.random.uniform(0, 1, (num_iters, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T)
    
    # A. LOCAL SECTION
    st.header(f"🎯 Impact Analysis: ZIP {zip_in}")
    idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]
    
    if len(idx) > 0:
        d = sim_results[:, idx[0]]
        m, s = norm.fit(d)
        
        # --- IMPACT TIER LOGIC ---
        if m < 0.15:
            tier, color = "LOW IMPACT", "#2ecc71" # Green
            desc = "This area currently meets most environmental and social baseline metrics."
        elif 0.15 <= m < 0.30:
            tier, color = "MEDIUM IMPACT", "#f1c40f" # Yellow
            desc = "This area shows emerging needs; a garden build would provide meaningful benefits."
        elif 0.30 <= m < 0.45:
            tier, color = "HIGH IMPACT", "#e67e22" # Orange
            desc = "Significant vulnerability identified. GKG builds here will have a major community effect."
        else:
            tier, color = "VERY HIGH IMPACT", "#e74c3c" # Red
            desc = "DANGER ZONE: This tract is in the highest tier of need for green space conversion."

        # High-Visibility Status Bar
        st.markdown(f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; margin-bottom:20px;">
                <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
                <p style="color:white; font-size:1.2rem; margin-top:10px; opacity:0.9;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.metric(label="Calculated Impact Score (Mean)", value=f"{m:.3f}", delta=f"± {s:.3f} SD")

        col_local_l, col_local_r = st.columns([2, 1])
        with col_local_l:
            fig_local, ax_local = plt.subplots(figsize=(10, 4.5))
            ax_local.hist(d, bins=30, color='#aed6f1', edgecolor='white', density=True, alpha=0.7)
            x_fit = np.linspace(min(d), max(d), 100)
            ax_local.plot(x_fit, norm.pdf(x_fit, m, s), color='#2e86c1', lw=3, label='Normal Distribution Fit')
            ax_local.axvline(m, color='#1b4f72', lw=2, label=f'Mean: {m:.3f}')
            ax_local.axvline(m + s, color='#e74c3c', ls='--', lw=1.5, label='+1 SD Bound')
            ax_local.axvline(m - s, color='#e74c3c', ls='--', lw=1.5, label='-1 SD Bound')
            ax_local.set_title("Probability Distribution of Impact Score")
            ax_local.legend(loc='upper right', fontsize='small')
            st.pyplot(fig_local)
        
        with col_local_r:
            st.markdown("### **Understanding the Graph**")
            st.write(f"""
            This graph shows the range of possible impact scores for ZIP **{zip_in}**.
            We ran 10,000 scenarios adjusting for different community priorities. 
            
            The **narrower** the curve, the more "certain" the impact is. 
            The **further right** the curve sits, the higher the need for a garden build.
            """)
            st.table(pd.DataFrame({
                "Metric": ["Avg Score", "Variance", "Confidence Range"],
                "Value": [f"{m:.4f}", f"{s:.4f}", f"{m-s:.3f} to {m+s:.3f}"]
            }))
    
    st.divider()

    # B. REGIONAL SECTION
    st.header("🌎 County-Wide Comparison")
    st.write("Where your selected ZIP stands compared to all 2,000+ census tracts in LA County.")
    
    median_scores = np.median(sim_results, axis=0)
    p25 = np.percentile(sim_results, 25, axis=0)
    p75 = np.percentile(sim_results, 75, axis=0)
    sorted_idx = np.argsort(median_scores)

    fig_reg, ax_reg = plt.subplots(figsize=(12, 3.5)) 
    ax_reg.fill_between(range(len(median_scores)), p25[sorted_idx], p75[sorted_idx], 
                        alpha=0.2, label='County IQR (25-75th)', color='#3498db')
    ax_reg.plot(median_scores[sorted_idx], color='#2980b9', lw=2, label='Median Trendline')
    ax_reg.set_ylabel("Impact Index")
    ax_reg.set_xlabel("Tracts Ranked by Need")
    ax_reg.grid(True, which='both', linestyle='--', alpha=0.3)
    ax_reg.legend(loc='lower right', fontsize='x-small')
    st.pyplot(fig_reg)

else:
    st.sidebar.error(f"ZIP {zip_in} not found. Please try another LA County ZIP.")

st.divider()

# ----------------------------
# 3. Deep-Dive & References
# ----------------------------

st.header("🔍 Detailed Factor Analysis")

def plot_component(df, col, name, unit, description, source_info, is_high_danger=True, bins=100):
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    danger_zone = df[df[col] > thresh] if is_high_danger else df[df[col] < thresh]
    zone_sym = ">" if is_high_danger else "<"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(description)
        st.caption(f"**Source:** {source_info}")
        st.table(pd.DataFrame({
            "Metric": ["Mean", "SD", "Danger Threshold", "Tracts in Danger"],
            "Details": [f"{mean_v:,.2f}", f"{std_v:,.2f}", f"{zone_sym} {thresh:,.2f}", str(len(danger_zone))]
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
        ax.plot(x, pdf * len(data) * bin_width, color='black', lw=1.5)
        ax.axvline(mean_v, color='black', label='Mean')
        ax.axvline(thresh, color='#e74c3c', ls='--', label='Danger Zone Cutoff')
        ax.set_xlabel(unit)
        ax.legend(fontsize='small')
        st.pyplot(fig)

# DATA CITATIONS FROM DELIVERABLE
ejsm_desc = "The **EJSM Score** evaluates hazard proximity, health risk, and social vulnerability. [cite: 20]"
ejsm_src = "USC & Occidental College Green Zones Program (2022) [cite: 20]"

income_desc = "**Median Household Income** tracks areas where financial barriers may prevent access to private green space. [cite: 49]"
income_src = "U.S. Census Bureau, Median HH Income ($93,525 Mean) [cite: 51]"

heat_desc = "**Heat Burden** (Degree Hour Days) measures urban heat island intensity. High-heat areas are critical targets. [cite: 108]"
heat_src = "Safe Clean Water Program (SCWP) LA [cite: 109]"

snap_desc = "**SNAP Participation** acts as a proxy for food insecurity in census tracts. [cite: 55]"
snap_src = "USDA / SNAP Participation Data (Representative Sample) [cite: 92]"

plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', ejsm_desc, ejsm_src, is_high_danger=False, bins=20)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', 'Dollars ($)', income_desc, income_src, is_high_danger=False, bins=250)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'Degree Hour Days', heat_desc, heat_src, is_high_danger=True, bins=150)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% of Population', snap_desc, snap_src, is_high_danger=True, bins=150)

st.sidebar.markdown("---")
st.sidebar.info("Developed for **Good Karma Gardens** to quantify community equity and sustainability. [cite: 218]")
