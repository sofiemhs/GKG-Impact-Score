import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- HEADER SECTION ---
st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")

with st.expander("📖 Methodology, Standardization & Instructions (Click to Expand)"):
    st.markdown("""
    ### **1. How to Use This Tool**
    * **Update Location:** Use the **Sidebar on the left** to enter any 5-digit Los Angeles County ZIP code.
    
    ### **2. How We Standardize the Data**
    To compare different units (like Dollars vs. Heat Degrees), we scale every factor between **0.0 and 1.0**:
    * **High Value (1.0) = High Need:** For Heat and SNAP, a higher raw number equals a higher score.
    * **Inverse Scaling:** For Income, a **lower** raw number equals a higher impact score.
    * **Formula:** $x_{std} = \\frac{x - min(x)}{max(x) - min(x)}$
    
    ### **3. Impact Tier Qualifications**
    * 🟢 **Low Impact (0.00 - 0.15):** High-resource areas.
    * 🟡 **Medium Impact (0.15 - 0.30):** Moderate need.
    * 🟠 **High Impact (0.30 - 0.45):** Significant vulnerability.
    * 🔴 **Very High Impact (0.45+):** **'Danger Zone'**; critical need for intervention.
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

    # Heat Burden - FILENAME CORRECTED HERE (Added the 's')
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
# 2. Main Analytics (Monte Carlo)
# ----------------------------

st.sidebar.header("📍 Change Location Here")
zip_in = st.sidebar.text_input("Enter ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if not match.empty:
    target_geoid = match.iloc[0]['GEOID10']
    
    # 10k Monte Carlo
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    num_iters = 10000 
    weights = np.random.uniform(0, 1, (num_iters, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T)
    
    st.header(f"🎯 Impact Analysis: ZIP {zip_in}")
    idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]
    
    if len(idx) > 0:
        d = sim_results[:, idx[0]]
        m, s = norm.fit(d)
        
        # Tier Logic
        if m < 0.15: tier, color, desc = "LOW IMPACT", "#2ecc71", "Area meets healthy baseline metrics."
        elif 0.15 <= m < 0.30: tier, color, desc = "MEDIUM IMPACT", "#f1c40f", "Emerging needs detected."
        elif 0.30 <= m < 0.45: tier, color, desc = "HIGH IMPACT", "#e67e22", "Significant vulnerability. High priority."
        else: tier, color, desc = "VERY HIGH IMPACT", "#e74c3c", "DANGER ZONE: Critical need."

        st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
            <p style="color:white; font-size:1.2rem; margin-top:10px;">{desc}</p></div>""", unsafe_allow_html=True)

        st.metric(label="Impact Score (The 'Need' Level)", value=f"{m:.3f}", delta=f"± {s:.3f} Variability")
        
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.hist(d, bins=30, color='#aed6f1', edgecolor='white', density=True, alpha=0.7)
            x_range = np.linspace(min(d), max(d), 100)
            ax.plot(x_range, norm.pdf(x_range, m, s), color='#2e86c1', lw=3, label='Monte Carlo Fit')
            ax.axvline(m, color='#1b4f72', lw=2, label=f'Mean: {m:.3f}')
            ax.axvline(m-s, color='#e74c3c', ls='--', lw=1.5, label=f'-1 SD: {m-s:.3f}')
            ax.axvline(m+s, color='#e74c3c', ls='--', lw=1.5, label=f'+1 SD: {m+s:.3f}')
            ax.set_title(f"Simulation Variance for ZIP {zip_in}")
            ax.set_xlabel("Potential Standardized Impact Score")
            ax.legend(fontsize='small')
            st.pyplot(fig)
        with col_r:
            st.markdown("### **Statistical Summary**")
            st.table(pd.DataFrame({
                "Metric": ["Avg Score", "Uncertainty (SD)", "Lower Bound (-1 SD)", "Upper Bound (+1 SD)"], 
                "Value": [f"{m:.4f}", f"{s:.4f}", f"{m-s:.4f}", f"{m+s:.4f}"]
            }))
            st.caption("**What are SD Bounds?** Standard Deviation (SD) shows how much the score changes when we prioritize different factors. 68% of scenarios fall between the Red Dashed lines.")

    st.divider()

    # --- RESTORED CDF CURVE ---
    st.header("🌎 County-Wide Impact Ranking")
    st.write("Higher on the curve means higher need relative to all other LA County tracts.")
    
    median_scores = np.median(sim_results, axis=0)
    p25 = np.percentile(sim_results, 25, axis=0)
    p75 = np.percentile(sim_results, 75, axis=0)
    sorted_idx = np.argsort(median_scores)

    fig_reg, ax_reg = plt.subplots(figsize=(12, 3.5)) 
    ax_reg.fill_between(range(len(median_scores)), p25[sorted_idx], p75[sorted_idx], alpha=0.2, color='#3498db', label='County IQR (25-75th)')
    ax_reg.plot(median_scores[sorted_idx], color='#2980b9', lw=2, label='County Median Trendline')
    
    # Red dot for current ZIP
    if len(idx) > 0:
        local_rank = np.where(sorted_idx == idx[0])[0][0]
        ax_reg.scatter(local_rank, m, color='red', s=100, zorder=5, label=f'ZIP {zip_in} Rank')

    ax_reg.set_title("LA County Cumulative Need Ranking (CDF)")
    ax_reg.set_ylabel("Impact Index")
    ax_reg.set_xlabel("Tracts Ranked by Need")
    ax_reg.legend(loc='lower right', fontsize='small')
    st.pyplot(fig_reg)

st.divider()

# ----------------------------
# 3. Factor Deep-Dive (Standard Deviation Curve Fit)
# ----------------------------

st.header("🔍 Factor Distributions & Danger Zones")
st.write("The **Red Areas** are the 'Danger Zones' (Top 16% of need).")



def plot_component(df, col, name, unit, description, source_info, is_high_danger=True, bins=100):
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(description)
        st.caption(f"**Source:** {source_info}")
        st.table(pd.DataFrame({
            "Metric": ["County Mean", "Std Deviation", "Danger Cutoff"], 
            "Value": [f"{mean_v:,.2f}", f"{std_v:,.2f}", f"{thresh:,.2f}"]
        }))
        st.markdown(f"**Interpretation:** Once a tract hits **{thresh:,.2f}**, it enters the 'Danger Zone' (Red).")

    with c2:
        fig, ax = plt.subplots(figsize=(10, 4))
        counts, bin_edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.8, density=True)
        for i in range(len(patches)):
            mid = (bin_edges[i] + bin_edges[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        x_axis = np.linspace(data.min(), data.max(), 500)
        ax.plot(x_axis, norm.pdf(x_axis, mean_v, std_v), color='black', lw=2, label='Normal Distribution')
        ax.axvline(mean_v, color='black', label='County Mean')
        ax.axvline(thresh, color='#e74c3c', ls='--', lw=2, label='Danger Threshold')
        ax.set_title(f"LA County Distribution: {name}")
        ax.set_xlabel(unit)
        ax.legend(fontsize='small')
        st.pyplot(fig)

# Factors
plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', "Pollution and vulnerability.", "USC/Occidental College (2022)", is_high_danger=False, bins=20)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', 'USD ($)', "Lower income = Higher need.", "U.S. Census Bureau", is_high_danger=False, bins=250)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'Deg-Hr Days', "Urban Heat Island intensity.", "SCWP LA", is_high_danger=True, bins=150)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% Pop', "Food insecurity proxy.", "USDA Data", is_high_danger=True, bins=150)
