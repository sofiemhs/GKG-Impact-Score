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

with st.expander("📖 Project Definitions and Glossary"):
    st.markdown("""
    ### **Key Terms**
    * **Impact Index:** A weighted standardization of environmental and socioeconomic factors used to estimate the relative impact of garden projects in neighborhoods[cite: 232].
    * **Danger Zone:** Census tracts classified as having the highest need, representing areas with lower access to environmental resources or higher socioeconomic vulnerability[cite: 234].
    * **Standardization:** The process of scaling measured factors (Min to Max) down to a uniform 0.0 to 1.0 range to allow for integrated weighting[cite: 605].
    * **Monte Carlo Simulation:** A statistical technique used to assess the stability of our model by running 10,000 iterations with random weighting coefficients[cite: 240, 610].
    """)

with st.expander("📝 Methodology, Data Sources and Research"):
    st.markdown(fr"""
    ### **Research Question**
    "What impact are Good Karma Gardens (GKG) builds having?"[cite: 562]. This dashboard identifies regions across Los Angeles County where additional green spaces provide the greatest social, economic, and environmental impact[cite: 437].

    ### **The Four Pillars of Impact**
    * **Environmental Justice (EJSM):** Evaluates hazard proximity, health risk, social/health vulnerability, and climate change vulnerability (tree canopy coverage)[cite: 575].
    * **Median Household Income:** Identifies higher-priority zones for investment where income falls below the county mean[cite: 389].
    * **Urban Heat (Degree Hours):** Measures cumulative heat exposure (degrees above 80°F) to identify zones where green space could provide cooling benefits[cite: 392, 599].
    * **Food Access (SNAP):** Tracks tracts where high SNAP participation indicates a need for green spaces that support food access and community resilience[cite: 391].

    ### **Robustness Index (RI)**
    To ensure our findings are reliable, we conducted a **Monte Carlo simulation** using 10,000 iterations to account for uncertainty in indicator weighting[cite: 394, 610].
    * **Mean Standard Deviation:** 0.0465 
    * **Mean Standard Error:** 0.000465 
    """, unsafe_allow_html=True)

# ----------------------------
# 1. Robust Data Loading 
# ----------------------------

@st.cache_data
def load_all_data():
    possible_paths = ["data", "GKG-Impact-Score/data"]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p): data_path = p; break
    if data_path is None: raise FileNotFoundError("Data folder not found.")

    def smart_load(filename):
        path = f"{data_path}/{filename}"
        # Fixing BadZipFile error: try reading as CSV if Excel fails
        try:
            return pd.read_excel(path, engine='openpyxl')
        except Exception:
            try: return pd.read_csv(path)
            except: raise Exception(f"Could not read {filename} as Excel or CSV.")

    # 1. EJSM (USC/Occidental College/LA County 2022) [cite: 575]
    df_ejsm = smart_load("EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Income (ACS 5-Year Estimates) [cite: 283]
    df_income = smart_load("Income_original.csv")
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Heat (Safe Clean Water Program LA) [cite: 283, 599]
    df_heat = smart_load("DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # 4. Food Access (SNAP/USDA) [cite: 283]
    df_snap = smart_load("Food_Deserts_CLEAN.csv")
    df_snap.columns = df_snap.columns.str.strip()
    def format_geoid(x):
        s = str(x).split('.')[0].strip()
        if len(s) <= 7: return "06037" + s.zfill(6)
        return s.zfill(11)
    df_snap['GEOID10'] = df_snap['CT10'].apply(format_geoid)
    df_snap['SNAP_pct'] = (pd.to_numeric(df_snap['TractSNAP'], errors='coerce') / pd.to_numeric(df_snap['Pop2010'], errors='coerce')) * 100
    df_snap = df_snap.dropna(subset=['SNAP_pct'])

    # 5. Zip-to-Tract Crosswalk
    df_ziptract = smart_load("ZIP_TRACT_122025.xlsx")
    df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
    df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str).str.split('.').str[0].str.zfill(11)

    # Standardization (0.0 to 1.0) [cite: 605]
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
# 2. LOCAL IMPACT ANALYSIS
# ----------------------------

st.sidebar.header("📍 Strategic Prioritization")
zip_in = st.sidebar.text_input("Enter ZIP Code in LA County:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if match.empty:
    st.error("ZIP Code not found in LA County datasets."); st.stop()

target_geoid = match.iloc[0]['GEOID10']
idx_row = df_comb[df_comb['GEOID10'] == target_geoid].index[0]
raw_scores = df_comb.iloc[idx_row][['s_e', 's_i', 's_h', 's_s']]
actual_score = 4 * (np.dot(raw_scores.values, weights_list)) / sum(weights_list)

# Tier Logic [cite: 415]
if actual_score < 1.0: tier, color = "LOW IMPACT", "#2ecc71"
elif 1.0 <= actual_score < 2.5: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 2.5 <= actual_score < 3.2: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "EXTREME IMPACT (DANGER ZONE)", "#e74c3c"

st.header("📊 Estimated Impact Score")
st.markdown(f"""
<div style="background-color:{color}; padding:25px; border-radius:15px; text-align:center;">
    <p style="color:white; font-size:1.3rem; margin:0; font-weight:bold;">{tier}</p>
    <h1 style="color:white; font-size:5.5rem; margin:0;">{actual_score:.2f}</h1>
    <p style="color:white; font-weight:bold;">Calculated Benefit for ZIP {zip_in}</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# 3. PILLAR DEEP DIVE (DANGER ZONES)
# ----------------------------
st.divider()
st.header("🔍 Indicator Danger Zones")

def plot_pillar(df, col, name, unit, score_key, thresh_val, is_high_bad=True, source="", calc=""):
    sub = df[df['GEOID10'] == target_geoid]
    if sub.empty: return

    val = sub[col].values[0]
    std_val = raw_scores[score_key]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.markdown(f"**Source:** {source}")
        st.markdown(f"**Calculation:** {calc}")
        st.metric("Raw Value", f"{val:,.1f} {unit}")
        st.metric("Standardized (0-1)", f"{std_val:.3f}")

        # Danger Zone Logic [cite: 578, 584, 590, 603]
        is_danger = (is_high_bad and val > thresh_val) or (not is_high_bad and val < thresh_val)
        if is_danger:
            st.error(f"🚨 **DANGER ZONE:** This area is beyond the project threshold of {thresh_val} {unit}.")
        else:
            st.success("✅ **STABLE:** This area is within the county average range.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(df[col].dropna(), bins=50, color='#bdc3c7', alpha=0.6)
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(thresh_val, color='red', ls=':', lw=2, label='Danger Threshold')
        ax.set_xlabel(unit); ax.legend(fontsize='x-small')
        st.pyplot(fig)
    st.divider()

# Pillars from Deliverables [cite: 575, 584, 590, 603]
pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Points', 's_e', 7.65, False, "USC/Occidental/LA County (2022)", "Hazard, Health, Social, Canopy [cite: 575]"),
    (df_income, 'med_hh_income', 'Median Household Income', '$USD', 's_i', 53423.09, False, "US Census Bureau", "Inverted scale: lower = higher impact [cite: 694]"),
    (df_heat, 'DegHourDay', 'Urban Heat Burden', 'Degree Hours', 's_h', 82.61, True, "SCWP LA", "Cumulative degrees above 80°F [cite: 599]"),
    (df_snap, 'SNAP_pct', 'Food Access (SNAP)', '% Pop', 's_s', 5.52, True, "USDA/Census", "SNAP participation rate [cite: 590]")
]

for p in pillars:
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], is_high_bad=p[6], source=p[7], calc=p[8])

# ----------------------------
# 4. WATER USAGE TOOL (SKELETON) [cite: 617, 630]
# ----------------------------
st.header("💧 Water Usage Dashboard")
st.markdown("""
This tool estimates the annual water and cost savings when conventional lawns are replaced with California native plants[cite: 244, 396].
""")

sq_ft = st.number_input("Enter square footage of lawn to convert:", min_value=0, value=500)
plant_type = st.selectbox("Select Native Plant Type:", ["Shrub", "Ground Cover", "Vine", "Bulb", "Succulent"])

# Methodology: ETC = ETo * Plant Factor [cite: 623]
eto = 48.34  # Annual Reference Evapotranspiration for Santa Monica [cite: 623]
pf_map = {"Shrub": 0.2, "Ground Cover": 0.2, "Vine": 0.2, "Bulb": 0.2, "Succulent": 0.05} # [cite: 622]
lawn_pf = 0.8  # Standard ornamental lawn factor

gallons_saved = (sq_ft * (eto * lawn_pf) * 0.623) - (sq_ft * (eto * pf_map[plant_type]) * 0.623)
st.metric("Estimated Annual Water Saved", f"{gallons_saved:,.0f} Gallons")
st.caption("Calculated using WUCOLS and CIMIS Santa Monica 2024 data[cite: 623].")
