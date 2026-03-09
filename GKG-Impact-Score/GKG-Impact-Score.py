import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- HEADER SECTION ---
st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")

with st.expander("📖 Methodology & Standardization (Click to Expand)"):
    st.markdown("""
    ### **1. How to Use This Tool**
    * **Update Location:** Use the **Sidebar on the left** to enter any 5-digit Los Angeles County ZIP code.
    
    ### **2. How We Standardize the Data**
    To compare different units, we scale every factor between **0.0 and 1.0**:
    * **High Value (1.0) = High Need.**
    * **Inverse Scaling:** For Income, a **lower** raw number equals a higher impact score.
    
    ### **3. Impact Tier Qualifications**
    * 🟢 **Low Impact (0.00 - 0.15):** High-resource areas.
    * 🟡 **Medium Impact (0.15 - 0.30):** Moderate need.
    * 🟠 **High Impact (0.30 - 0.45):** Significant vulnerability.
    * 🔴 **Very High Impact (0.45+):** **'Danger Zone'**; critical need.
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

    # Data Loading
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',','')
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'], errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat = df_heat.dropna(subset=['DegHourDay'])
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    df_snap = pd.read_csv(f"{data_path}/Food_Deserts (1).csv")
    df_snap.columns = df_snap.columns.str.strip()
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0) & (df_snap['TractSNAP'] != 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)

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

st.sidebar.header("📍 Change Location")
zip_in = st.sidebar.text_input("Enter ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if match.empty:
    st.error(f"❌ ZIP Code {zip_in} Not Found in LA County Data.")
    st.stop()

target_geoid = match.iloc[0]['GEOID10']
idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]

if len(idx) > 0:
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    weights = np.random.uniform(0, 1, (10000, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T)
    
    d = sim_results[:, idx[0]]
    m, s = norm.fit(d)
    
    # DRIVER ANALYSIS
    raw_scores = df_comb.iloc[idx[0]][['s_e', 's_i', 's_h', 's_s']]
    total_raw = raw_scores.sum()
    pcts = (raw_scores / total_raw) if total_raw > 0 else raw_scores
    
    labels = {
        's_e': 'Environmental Justice (EJSM)',
        's_i': 'Economic Need (Income)',
        's_h': 'Heat Burden',
        's_s': 'Food Insecurity (SNAP)'
    }
    
    drivers = [labels[k] for k, v in pcts.items() if v >= 0.30]
    driver_text = f" & driven primarily by **{', '.join(drivers)}**" if drivers else ""

    if m < 0.15: tier, color, desc = "LOW IMPACT", "#2ecc71", "Healthy baseline."
    elif 0.15 <= m < 0.30: tier, color, desc = "MEDIUM IMPACT", "#f1c40f", "Emerging needs detected."
    elif 0.30 <= m < 0.45: tier, color, desc = "HIGH IMPACT", "#e67e22", "Significant vulnerability."
    else: tier, color, desc = "VERY HIGH IMPACT", "#e74c3c", "DANGER ZONE: Critical need."

    st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
        <p style="color:white; font-size:1.4rem; margin-top:5px; font-weight:bold;">{desc}{driver_text}</p></div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.hist(d, bins=30, color='#aed6f1', density=True, alpha=0.7)
        x_range = np.linspace(min(d), max(d), 100)
        ax.plot(x_range, norm.pdf(x_range, m, s), color='#2e86c1', lw=3)
        ax.axvline(m, color='#1b4f72', lw=2, label=f'Mean: {m:.3f}')
        ax.axvline(m-s, color='#e74c3c', ls='--', label='-1 SD')
        ax.axvline(m+s, color='#e74c3c', ls='--', label='+1 SD')
        ax.set_title(f"Impact Simulation for ZIP {zip_in}")
        ax.legend()
        st.pyplot(fig)
    with col_r:
        st.markdown("### **Statistical Breakdown**")
        st.table(pd.DataFrame({
            "Metric": ["Avg Impact Score", "Uncertainty (SD)", "Lower Bound", "Upper Bound"], 
            "Value": [f"{m:.4f}", f"{s:.4f}", f"{m-s:.4f}", f"{m+s:.4f}"]
        }))

# ----------------------------
# 3. Factor Deep-Dive (FIXED)
# ----------------------------
st.divider()
st.header("🔍 Factor Distributions & Danger Zones")

def plot_component(df, col, name, unit, description, source_info, score_key, is_high_danger=True):
    # Safety Check for missing local data
    local_data = df[df['GEOID10'] == target_geoid]
    if local_data.empty:
        st.warning(f"No specific {name} data available for this tract.")
        return

    current_val = local_data[col].values[0]
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    
    # FIXED: Use explicit score_key instead of string slicing logic
    std_score = df_comb.iloc[idx[0]][score_key]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(description)
        st.metric(f"Current Value ({zip_in})", f"{current_val:,.2f} {unit}")
        st.write(f"**Impact Weight:** This factor contributes **{std_score:.3f}** to the final standardized score.")
        st.caption(f"**Source:** {source_info}")

    with c2:
        fig, ax = plt.subplots(figsize=(10, 4))
        counts, bin_edges, patches = ax.hist(data, bins=100, color='#bdc3c7', alpha=0.8, density=True)
        for i in range(len(patches)):
            mid = (bin_edges[i] + bin_edges[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        ax.axvline(current_val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(thresh, color='#e74c3c', ls='--', label='Danger Cutoff')
        ax.set_title(f"LA County Distribution: {name}")
        ax.legend()
        st.pyplot(fig)

# Component Plots - Passing explicit score_key
plot_component(df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Score', "Pollution and vulnerability.", "USC (2022)", 's_e', False)
st.divider()
plot_component(df_income, 'med_hh_income', 'Median HH Income', 'USD ($)', "Lower income = higher need.", "Census Bureau", 's_i', False)
st.divider()
plot_component(df_heat, 'DegHourDay', 'Heat Burden', 'Deg-Hr Days', "Heat island intensity.", "SCWP LA", 's_h', True)
st.divider()
plot_component(df_snap, 'SNAP_pct', 'Food Access (SNAP %)', '% Pop', "Food insecurity proxy.", "USDA Data", 's_s', True)
