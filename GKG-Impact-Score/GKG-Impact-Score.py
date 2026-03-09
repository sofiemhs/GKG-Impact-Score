import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Impact Score Analysis")

with st.expander("📖 Methodology, Data Sources & Years"):
    st.markdown("""
    ### **The Question We Are Answering**
    "What impact are Good Karma Gardens work converting space into gardens having based on location?"

    ### **Why Each Pillar Matters**
    * **[Environmental Justice (EJSM)](#environmental-justice-ejsm):** Communities will low Enviromental Justice score are at a higher need for increased green space
    * **[Economic Need](#economic-need):** Low Income communities are often overlooked and underserved for publically accessable green spaces
    * **[Heat Burden](#heat-burden):** Urban heat is a result of lack of canopy cover and can causes dangerously high temperatures, yet gardens can limit this effect by actively cooling these areas through transpiration.
    * **[Food Access (SNAP)](#food-access-snap):** Pinpoints 'food deserts' where affordable, fresh produce is scarce. Good karma Garden's work can help to aliviate this burden.

    ### **Standardization Logic**
    Every raw data point (dollars, degrees, or percentages) is standardized on a scale of **0.0 to 1.0**. 
    * **0.0** represents the lowest need/impact in the county.
    * **1.0** represents the highest need/impact in the county.
    * The total **Impact Score (0.0 - 4.0)** is the sum of these four pillars.
    """, unsafe_allow_html=True)

# ----------------------------
# 1. Data Loading 
# ----------------------------

@st.cache_data
def load_all_data():
    possible_paths = ["data", "GKG-Impact-Score/data"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p): data_path = p; break
    
    if data_path is None: raise FileNotFoundError("Data folder not found.")

    # 1. EJSM
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Income
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Heat
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # 4. Food Access (CORRECTED MATCHING LOGIC)
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts_CLEAN.csv")
    df_snap.columns = df_snap.columns.str.strip()
    
    # Clean the CT10 to ensure it's a string and padded with LA County FIPS if needed
    def format_geoid(x):
        s = str(x).split('.')[0].strip()
        if len(s) <= 7: # It's a 6-digit tract, add CA (06) + LA (037)
            return "06037" + s.zfill(6)
        return s.zfill(11)

    df_snap['GEOID10'] = df_snap['CT10'].apply(format_geoid)
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100

    # 5. Zip-to-Tract Crosswalk
    df_ziptract = pd.read_excel(f"{data_path}/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization helper
    def std(df, col, inv=False):
        s = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return 1 - s if inv else s

    df_ejsm['s'] = std(df_ejsm, 'CIscore')
    df_income['s'] = std(df_income, 'med_hh_income', inv=True)
    df_heat['s'] = std(df_heat, 'DegHourDay')
    df_snap['s'] = std(df_snap, 'SNAP_pct')

    # Merge all into master set
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

# Monte Carlo: 10,000 simulations
x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
sim_weights = np.random.uniform(0, 1, (10000, 4))
sim_weights /= sim_weights.sum(axis=1, keepdims=True)
sim_results = np.dot(sim_weights, x_matrix.T) 
local_sims = sim_results[:, idx_row] * 4
m_loc, s_loc = norm.fit(local_sims)

# Impact Range Logic
if actual_score < 0.8: tier, color = "LOW IMPACT", "#2ecc71"
elif 0.8 <= actual_score < 1.6: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 1.6 <= actual_score < 2.4: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "EXTREME IMPACT (DANGER ZONE)", "#e74c3c"

st.header("📊 Impact Score Approximation")

st.markdown(f"""
<div style="background-color:{color}; padding:20px; border-radius:15px; border: 2px solid rgba(0,0,0,0.1); margin-bottom:20px; text-align:center;">
    <p style="color:white; font-size:1.2rem; margin:0; font-weight:bold; text-transform:uppercase;">{tier}</p>
    <h1 style="color:white; font-size:5rem; margin:0; line-height:1;">{m_loc:.2f} <span style="font-size:1.5rem;">± {s_loc:.3f}</span></h1>
    <p style="color:white; font-weight:bold; margin-top:10px;">Calculated Impact Score for ZIP {zip_in}</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# 3. PILLAR DEEP-DIVE (VERIFIED LOGIC)
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep-Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True, source="", anchor_id=""):
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
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
        st.markdown(f"**Description:** {desc}")
        st.markdown(f"**Data Source:** {source}")
        st.metric(f"ZIP {zip_in} Raw Value", f"{val:,.1f} {unit}")
        st.metric("Standardized Score Contribution", f"{std_val:.3f} / 1.0")
        
        thresh = mean_v + std_v if is_high_danger else mean_v - std_v
        if (is_high_danger and val > thresh) or (not is_high_danger and val < thresh):
            st.error("🚨 **DANGER ZONE:** Metric exceeds critical threshold.")
        else:
            st.success("✅ **NORMAL RANGE:** Metric is within acceptable bounds.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        counts, edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        thresh_line = mean_v + std_v if is_high_danger else mean_v - std_v
        for i in range(len(patches)):
            mid = (edges[i] + edges[i+1]) / 2
            if (is_high_danger and mid > thresh_line) or (not is_high_danger and mid < thresh_line):
                patches[i].set_facecolor('#e74c3c')
        
        x_vals = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_vals, norm.pdf(x_vals, mean_v, std_v), color='black', lw=2, label='Normal Distribution')
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)
    st.divider()

pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Points', 
     "Evaluates hazard proximity, health risk, and canopy cover. Score 4-20. Threshold: < 7.65.", 
     's_e', 20, False, "USC / Occidental College (2022)", "environmental-justice-ejsm"),
    
    (df_income, 'med_hh_income', 'Economic Need', '$USD', 
     "Identifies income barriers. Threshold: < $53,423.", 
     's_i', 250, False, "US Census Bureau (2021)", "economic-need"),
    
    (df_heat, 'DegHourDay', 'Heat Burden', 'Days', 
     "Measures urban heat intensity. Threshold: > 82.61 Degree Hours.", 
     's_h', 150, True, "Safe Clean Water Program LA (2022)", "heat-burden"),
    
    (df_snap, 'SNAP_pct', 'Food Access (SNAP)', '% Pop', 
     "SNAP participation rate. Threshold: > 5.52%.", 
     's_s', 150, True, "USDA Food Access Research Atlas (2019)", "food-access-snap")
]

for p in sorted(pillars, key=lambda x: raw_scores[x[5]], reverse=True):
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], source=p[8], anchor_id=p[9])
