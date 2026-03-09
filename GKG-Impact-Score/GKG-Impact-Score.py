import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Precision Impact Analysis")

with st.expander("📖 Methodology, Data Sources & Years"):
    st.markdown("""
    | Pillar | Source | Year | Why it matters |
    | :--- | :--- | :--- | :--- |
    | **Env. Justice (EJSM)** | SCAG / CalEnviroScreen | 2022 | Identifies cumulative pollution burden. |
    | **Economic Need** | ACS 5-Year Estimates | 2021 | Prioritizes investment in resource-strapped areas. |
    | **Heat Burden** | NOAA / Urban Heat Watch | 2022 | Targets locations for cooling via transpiration. |
    | **Food Access (SNAP)** | USDA Food Access Research Atlas | 2019 | Maps food insecurity and 'food deserts.' |
    """)
    st.latex(r"Total Impact (0.0 - 4.0) = EJSM_{std} + Income_{std} + Heat_{std} + Food_{std}")
    st.info("Note: If data is missing for a tract, that pillar is assumed to be 0 for the final score.")

# ----------------------------
# 1. Data Loading & Standardizing
# ----------------------------

@st.cache_data
def load_all_data():
    possible_paths = ["data", "GKG-Impact-Score/data"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p): data_path = p; break
    
    if data_path is None: raise FileNotFoundError("Data folder not found.")

    # Load Dataframes
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    df_snap = pd.read_csv(f"{data_path}/Food_Deserts (1).csv")
    df_snap.columns = df_snap.columns.str.strip()
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100
    df_snap['GEOID10'] = df_snap['CT10'].astype(str).str.split('.').str[0].str.zfill(11)

    df_ziptract = pd.read_excel(f"{data_path}/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization
    def std(df, col, inv=False):
        s = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return 1 - s if inv else s

    df_ejsm['s'] = std(df_ejsm, 'CIscore')
    df_income['s'] = std(df_income, 'med_hh_income', inv=True)
    df_heat['s'] = std(df_heat, 'DegHourDay')
    df_snap['s'] = std(df_snap, 'SNAP_pct')

    df_comb = df_ejsm[['GEOID10','s']].merge(df_income[['GEOID10','s']], on='GEOID10', how='outer', suffixes=('_e','_i')) \
              .merge(df_heat[['GEOID10','s']], on='GEOID10', how='outer') \
              .merge(df_snap[['GEOID10','s']], on='GEOID10', how='outer', suffixes=('_h','_s')).fillna(0)
    
    return df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb

df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb = load_all_data()

# ----------------------------
# 2. LOCAL ANALYSIS 
# ----------------------------

st.sidebar.header("📍 Search Area")
zip_in = st.sidebar.text_input("Enter ZIP Code:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if match.empty:
    st.error("ZIP Code not found.")
    st.stop()

target_geoid = match.iloc[0]['GEOID10']
idx_row = df_comb[df_comb['GEOID10'] == target_geoid].index[0]
raw_scores = df_comb.iloc[idx_row][['s_e','s_i','s_h','s_s']]
actual_score = raw_scores.sum()

# Monte Carlo (Local Stats)
x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
sim_weights = np.random.uniform(0, 1, (1000, 4))
sim_weights /= sim_weights.sum(axis=1, keepdims=True)
sim_results = np.dot(sim_weights, x_matrix.T) 

local_sims = sim_results[:, idx_row] * 4
m_loc, s_loc = norm.fit(local_sims)

# Status Banner
if actual_score < 0.8: tier, color = "LOW IMPACT", "#2ecc71"
elif 0.8 <= actual_score < 1.6: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 1.6 <= actual_score < 2.4: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "VERY HIGH IMPACT", "#e74c3c"

labels = {'s_e': 'EJSM', 's_i': 'Economic Need', 's_h': 'Heat Burden', 's_s': 'Food Access'}
drivers = [labels[k] for k, v in (raw_scores / actual_score).items() if v >= 0.30] if actual_score > 0 else []

st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
    <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
    <p style="color:white; font-size:1.4rem; margin-top:5px; font-weight:bold;">Score is driven by {', '.join(drivers)}</p></div>""", unsafe_allow_html=True)

# Sensitivity Table
st.header(f"📊 Sensitivity & Error Analysis")
col_l, col_r = st.columns([2, 1])
with col_l:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(local_sims, bins=30, color='#aed6f1', density=True, alpha=0.7)
    x_range = np.linspace(min(local_sims), max(local_sims), 100)
    ax.plot(x_range, norm.pdf(x_range, m_loc, s_loc), color='#2e86c1', lw=3)
    ax.axvline(actual_score, color='#1b4f72', lw=3, label=f'Score: {actual_score:.2f}')
    ax.set_title(f"Score Variance Simulation for {zip_in}")
    ax.legend(fontsize='x-small')
    st.pyplot(fig)
with col_r:
    st.subheader("Statistical Breakdown")
    st.table(pd.DataFrame({
        "Metric": ["Calculated Impact", "Simulation Mean", "Volatility (SD)", "Lower Bound (-1SD)", "Upper Bound (+1SD)"],
        "Value": [f"{actual_score:.3f}", f"{m_loc:.3f}", f"{s_loc:.3f}", f"{actual_score-s_loc:.2f}", f"{actual_score+s_loc:.2f}"]
    }))

# ----------------------------
# 3. COUNTY CONTEXT (FIXED CDF)
# ----------------------------
st.divider()
st.header("🌎 County-Wide Impact Ranking")

# Fixing the CDF to match Monte_Carlo.png
medians = np.median(sim_results, axis=0)
p25 = np.percentile(sim_results, 25, axis=0)
p75 = np.percentile(sim_results, 75, axis=0)
sort_idx = np.argsort(medians)

mean_sd = np.std(sim_results, axis=0).mean()
mean_se = (np.std(sim_results, axis=0) / np.sqrt(1000)).mean()

fig_cdf, ax_cdf = plt.subplots(figsize=(12, 6))
ax_cdf.plot(medians[sort_idx], color='#1f77b4', lw=2.5, label='Median')
ax_cdf.fill_between(range(len(medians)), p25[sort_idx], p75[sort_idx], color='#1f77b4', alpha=0.2, label='25th-75th Percentile')

# Visual styling to match image
ax_cdf.grid(True, linestyle='-', alpha=0.2)
ax_cdf.set_title("Monte Carlo Simulation of Weighted Index\nwith Standard Deviation and Standard Error", fontsize=14)
ax_cdf.set_ylabel("Weighted Standardized Index", fontsize=12)
ax_cdf.set_xlabel("Census Tracts (Sorted)", fontsize=12)

# Stats Box
textstr = f"Mean SD = {mean_sd:.4f}\nMean SE = {mean_se:.6f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='gray')
ax_cdf.text(0.02, 0.95, textstr, transform=ax_cdf.transAxes, fontsize=10, verticalalignment='top', bbox=props)

ax_cdf.legend(loc='lower right')
st.pyplot(fig_cdf)

# ----------------------------
# 4. PILLAR DEEP-DIVE
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep-Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True):
    sub = df[df['GEOID10'] == target_geoid]
    if sub.empty:
        st.warning(f"⚠️ **DATA MISSING:** No local reporting for **{name}**. Standardized Score = **0.0**.")
        st.divider(); return

    val = sub[col].values[0]
    std_val = raw_scores[score_key]
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.write(desc)
        st.metric(f"ZIP {zip_in} Raw Value", f"{val:,.1f} {unit}")
        st.metric("Standardized Score (for Impact)", f"{std_val:.3f} / 1.0") # Point 1
        
        thresh = mean_v + std_v if is_high_danger else mean_v - std_v
        if (is_high_danger and val > thresh) or (not is_high_danger and val < thresh):
            st.error("🚨 **DANGER ZONE:** Metric exceeds critical threshold.")
        else:
            st.success("✅ **NORMAL RANGE:** Metric is within acceptable bounds.")

        st.table(pd.DataFrame({
            "Metric": ["County Mean", "+1 SD (Danger)", "-1 SD"],
            "Value": [f"{mean_v:,.2f}", f"{mean_v+std_v:,.2f}", f"{mean_v-std_v:,.2f}"]
        }))
    with col2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(mean_v + std_v, color='red', ls=':', lw=2, label='+1 SD')
        ax.axvline(mean_v - std_v, color='red', ls=':', lw=2, label='-1 SD')
        ax.legend(fontsize='xx-small', ncol=3)
        st.pyplot(fig)
    st.divider()

pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice', 'Points', "Pollution index.", 's_e', 20, False),
    (df_income, 'med_hh_income', 'Median HH Income', '$USD', "Economic metric.", 's_i', 250, False),
    (df_heat, 'DegHourDay', 'Heat Burden', 'Days', "Urban heat intensity.", 's_h', 150, True),
    (df_snap, 'SNAP_pct', 'Food Access', '% Pop', "Food sovereignty proxy.", 's_s', 150, True)
]

for p in sorted(pillars, key=lambda x: raw_scores[x[5]], reverse=True):
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
