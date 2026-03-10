import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 0: GLOBAL WEIGHT CONFIGURATION ---
w_e = 1.0  # EJSM
w_i = 1.0  # Income
w_h = 1.0  # Heat
w_s = 1.0  # SNAP
weights_list = [w_e, w_i, w_h, w_s]

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Impact Score Analysis")

with st.expander("📖 Definitions and Glossary"):
    st.markdown("""
    ### **Key Terms**
    * **Pillar Score Contribution:** The individual standardized score (0.0 to 1.0) of a category after weight application[cite: 544].
    * **% of Total Impact Score:** The influence a single pillar has on the final 4.0 score.
    * **Standardization:** Scaling raw data (dollars, degrees, or percentages) to a uniform 0.0 to 1.0 range using Min and Max values[cite: 544].
    * **Danger Zone:** Census tracts identified as having the highest need, typically defined as being more than one standard deviation (1 SD) beyond the mean[cite: 444, 472, 530].
    """)

with st.expander("📝 Methodology, Data Sources and Years"):
    st.markdown(fr"""
    ### **The Research Question**
    "What impact are Good Karma Gardens (GKG) builds having?"[cite: 417]. This dashboard identifies areas where community gardens would provide the most significant benefit by analyzing where socio-economic and environmental stressors overlap[cite: 418, 420].

    ### **Why Each Pillar Matters**
    * **Environmental Justice (EJSM):** Evaluates environmental justice across hazard proximity, health risk, social/health vulnerability, and climate change vulnerability (tree canopy)[cite: 433, 434, 440, 441].
    * **Median Household Income:** Lower income tracts often lack private green space; this tool identifies communities with the highest financial barriers to garden access[cite: 462, 629].
    * **Urban Heat:** Measured in **Degree Hours per day**, identifying areas where air temperature frequently exceeds 80°F, highlighting a lack of cooling canopy cover[cite: 522, 523, 530].
    * **Food Access (SNAP):** Pinpoints areas with high SNAP participation rates, identifying "food deserts" where affordable, fresh produce is scarce[cite: 468, 472].

    ### **Standardization & Math**
    We scaled all measured factors to a range of **0.0 to 1.0**[cite: 544]. To account for variability in subjective weighting, we utilized a **Monte Carlo simulation** with 10,000 iterations to find the possible error for our equation[cite: 546, 549].
    
    **Impact Score Equation:**
    $$Impact Score = 4 \times \frac{{\sum (w_{{i}} \times s_{{i}})}}{{\sum w_{{i}}}}$$
    
    **Simulation Statistics:**
    * **Mean Standard Deviation:** 0.0465 [cite: 550]
    * **Mean Standard Error:** 0.000465 [cite: 550]
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

    # 1. EJSM - Adopted by LA County Green Zones Program [cite: 433]
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Income - Mean: $93,525.12 [cite: 464]
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Heat - Safe Clean Water Program LA [cite: 522]
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # 4. Food Access - SNAP [cite: 468]
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts_CLEAN.csv")
    df_snap.columns = df_snap.columns.str.strip()
    def format_geoid(x):
        s = str(x).split('.')[0].strip()
        if len(s) <= 7: return "06037" + s.zfill(6)
        return s.zfill(11)
    df_snap['GEOID10'] = df_snap['CT10'].apply(format_geoid)
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100

    # 5. Zip Crosswalk
    df_ziptract = pd.read_excel(f"{data_path}/ZIP_TRACT_122025.xlsx", engine='openpyxl')
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

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
zip_in = st.sidebar.text_input("Enter ZIP Code in LA County:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

ERROR_MSG = "The ZIP Code either does not exist in LA County or data is missing. Please try another."

if match.empty:
    st.error(ERROR_MSG); st.stop()

target_geoid = match.iloc[0]['GEOID10']
if target_geoid not in df_comb['GEOID10'].values:
    st.error(ERROR_MSG); st.stop()

idx_row = df_comb[df_comb['GEOID10'] == target_geoid].index[0]
raw_scores = df_comb.iloc[idx_row][['s_e', 's_i', 's_h', 's_s']]

actual_score = 4 * ( (raw_scores['s_e'] * w_e) + (raw_scores['s_i'] * w_i) + (raw_scores['s_h'] * w_h) + (raw_scores['s_s'] * w_s) ) / sum(weights_list)

# Tier Logic based on overall need [cite: 421]
if actual_score < 1.0: tier, color = "LOW IMPACT", "#2ecc71"
elif 1.0 <= actual_score < 2.0: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 2.0 <= actual_score < 3.0: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "EXTREME IMPACT (DANGER ZONE)", "#e74c3c"

st.header("📊 Impact Score Approximation")

st.markdown(f"""
<div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center;">
    <p style="color:white; font-size:1.2rem; margin:0; font-weight:bold;">{tier}</p>
    <h1 style="color:white; font-size:5rem; margin:0;">{actual_score:.2f}</h1>
    <p style="color:white; font-weight:bold;">Estimated Impact for ZIP {zip_in}</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# 3. PILLAR DEEP DIVE
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, weight, thresh_val, is_high_bad=True, source="", calc_expl=""):
    sub = df[df['GEOID10'] == target_geoid]
    if sub.empty: return

    val = sub[col].values[0]
    std_val = raw_scores[score_key]
    data = df[col].dropna()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.markdown(f"**Data Source:** {source}")
        st.markdown(f"**Calculation:** {calc_expl}")
        st.metric(f"Raw Value", f"{val:,.1f} {unit}")
        st.metric("Pillar Score", f"{std_val:.3f} / 1.0")

        if (is_high_bad and val > thresh_val) or (not is_high_bad and val < thresh_val):
            st.error(f"🚨 **DANGER ZONE:** This tract is beyond the threshold of {thresh_val} {unit}[cite: 444].")
        else:
            st.success("✅ **STABLE:** This metric is within the county average range.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7)
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(thresh_val, color='red', ls=':', lw=2, label='Danger Threshold')
        ax.set_xlabel(unit); ax.legend(fontsize='x-small')
        st.pyplot(fig)
    st.divider()

# Thresholds from Project Report [cite: 444, 464, 472, 530]
pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Points', 
     "Evaluates hazard and vulnerability[cite: 433].", 's_e', 20, w_e, 7.65, False, 
     "USC / Occidental College (2022)", "Standardized 4-20 scale[cite: 442]."),
    
    (df_income, 'med_hh_income', 'Median Household Income', '$USD', 
     "Identifies underserved communities[cite: 462].", 's_i', 100, w_i, 53423.09, False, 
     "US Census Bureau", "Inverted; lower income = higher need[cite: 464]."),
    
    (df_heat, 'DegHourDay', 'Heat Burden', 'Degree Hours', 
     "Excess heat above 80°F[cite: 522].", 's_h', 50, w_h, 82.61, True, 
     "Safe Clean Water Program LA", "Median: 42.36[cite: 525, 530]."),
    
    (df_snap, 'SNAP_pct', 'Food Access (SNAP)', '% Pop', 
     "Identifies 'food deserts'[cite: 468].", 's_s', 50, w_s, 5.52, True, 
     "USDA Food Access Research Atlas", "Mean usage rate: 3.15%[cite: 471, 472].")
]

for p in pillars:
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], is_high_bad=p[9], source=p[10], calc_expl=p[11])
