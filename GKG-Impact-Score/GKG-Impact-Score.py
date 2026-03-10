import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 0: GLOBAL WEIGHT CONFIGURATION ---
w_e = 1.0  # Environmental Justice (EJSM)
w_i = 1.0  # Median Household Income
w_h = 1.0  # Heat Burden (Temperature)
w_s = 1.0  # Food Access (SNAP)
weights_list = [w_e, w_i, w_h, w_s]

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Quantifying Green Space Impact")

with st.expander("📖 Definitions and Glossary"):
    st.markdown("""
    ### **Key Terms**
    * **Impact Index:** A weighted standardization of environmental and socioeconomic factors used to estimate the relative impact of garden projects in neighborhoods[cite: 158].
    * **Pillar Score Contribution:** The individual standardized score (0.0 to 1.0) of a category after weight application, representing the "raw" strength of that specific community need[cite: 379].
    * **% of Total Impact Score:** The influence a single pillar has on the final 4.0 score, highlighting which stressors (e.g., heat vs. income) drive local vulnerability[cite: 158].
    * **Standardization:** The process of scaling raw units (dollars, degrees, or percentages) to a uniform 0.0 to 1.0 range based on county-wide Min/Max values[cite: 379].
    * **Danger Zone:** Census tracts classified as having the highest need, typically defined as being more than one standard deviation (1 SD) beyond the county mean[cite: 160, 448].
    * **Robustness Index (RI):** A measure of model stability calculated via Monte Carlo simulations to ensure results are reliable despite weighting uncertainties[cite: 168].
    """)

with st.expander("📝 Methodology, Data Sources and Research"):
    st.markdown(fr"""
    ### **The Research Question**
    "What impact are Good Karma Gardens (GKG) builds having?"[cite: 421]. This dashboard identifies regions across Los Angeles County where community gardens provide the greatest social and environmental equity by converting conventional lawns into native plant spaces[cite: 153, 157].

    ### **Why Each Pillar Matters**
    * **Environmental Justice (EJSM):** Evaluates hazard proximity, health risk, social/health vulnerability, and climate change vulnerability (tree canopy) [cite: 437-445]. Developed by USC/Occidental College and adopted by the LA County Green Zones Program[cite: 437].
    * **Median Household Income:** Identifies higher-priority zones for investment where financial barriers limit access to private or public green spaces[cite: 163, 466].
    * **Urban Heat (Degree Hours):** Measures cumulative heat exposure (degrees above 80°F) to pinpoint "high heat burden zones" where vegetation could provide critical cooling[cite: 166, 373].
    * **Food Access (SNAP):** Tracks tracts with high SNAP participation to identify "food deserts" where gardens can support community resilience and fresh produce access[cite: 165, 472].

    ### **Standardization & Impact Calculation**
    Every data point is standardized from **0.0 (lowest need)** to **1.0 (highest need)**[cite: 379]. The total **Impact Score (0.0 to 4.0)** uses a weighted average:
    
    $$Impact Score = 4 \times \frac{{\sum (w_{{i}} \times s_{{i}})}}{{\sum w_{{i}}}}$$
    
    **Simulation Statistics (10,000 Iterations):**
    * **Mean Standard Deviation:** 0.0465 [cite: 385]
    * **Mean Standard Error:** 0.000465 [cite: 385]

    ### **Impact Ranges and Severity Logic**
    * <span style="color:#2ecc71; font-weight:bold;">0.0 to 0.8 (Low Impact):</span> Minimal environmental or social stressors.
    * <span style="color:#f1c40f; font-weight:bold;">0.8 to 1.6 (Medium Impact):</span> Average county-wide need levels.
    * <span style="color:#e67e22; font-weight:bold;">1.6 to 2.4 (High Impact):</span> High vulnerability; significant benefit from greening.
    * <span style="color:#e74c3c; font-weight:bold;">2.4 to 4.0 (Extreme Impact/Danger Zone):</span> Overlapping extreme stressors requiring urgent intervention[cite: 160, 177].
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

    # 1. EJSM - Adopted by LA County Green Zones Program
    df_ejsm = pd.read_csv(f"{data_path}/EJSM_Origonal.csv")
    df_ejsm.columns = df_ejsm.columns.str.strip()
    df_ejsm['GEOID10'] = df_ejsm['Tract_1'].astype(str).str.split('.').str[0].str.zfill(11)
    df_ejsm['CIscore'] = pd.to_numeric(df_ejsm['CIscore'], errors='coerce')
    df_ejsm = df_ejsm.dropna(subset=['CIscore'])

    # 2. Income - County Mean: $93,525.12
    df_income = pd.read_csv(f"{data_path}/Income_original.csv")
    df_income['med_hh_income'] = pd.to_numeric(df_income['med_hh_income'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')
    df_income = df_income[df_income['med_hh_income'].notna() & (df_income['med_hh_income'] != 0)]
    df_income['GEOID10'] = df_income['tract'].astype(str).str.split('.').str[0].str.zfill(11)

    # 3. Heat - Safe Clean Water Program LA
    df_heat = pd.read_csv(f"{data_path}/DegHourDays_Original.csv")
    df_heat.columns = df_heat.columns.str.strip()
    df_heat['DegHourDay'] = pd.to_numeric(df_heat['DegHourDay'], errors='coerce')
    df_heat['GEOID10'] = df_heat['FIPS'].astype(str).str.split('.').str[0].str.zfill(11)

    # 4. Food Access - SNAP/USDA Research Atlas
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts_CLEAN.csv")
    df_snap.columns = df_snap.columns.str.strip()
    
    def format_geoid(x):
        s = str(x).split('.')[0].strip()
        if len(s) <= 7:
            return "06037" + s.zfill(6)
        return s.zfill(11)

    df_snap['GEOID10'] = df_snap['CT10'].apply(format_geoid)
    df_snap['TractSNAP'] = pd.to_numeric(df_snap['TractSNAP'], errors='coerce')
    df_snap['Pop2010'] = pd.to_numeric(df_snap['Pop2010'], errors='coerce')
    df_snap = df_snap[(df_snap['Pop2010'] > 0)].dropna(subset=['TractSNAP'])
    df_snap['SNAP_pct'] = (df_snap['TractSNAP'] / df_snap['Pop2010']) * 100

    # 5. Zip to Tract Crosswalk
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

st.sidebar.header("📍 Strategic Prioritization")
zip_in = st.sidebar.text_input("Enter ZIP Code in LA County:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

ERROR_MSG = "The ZIP Code either does not exist within Los Angeles County or lacks reliable reporting data. Please try another ZIP."

if match.empty:
    st.error(ERROR_MSG); st.stop()

target_geoid = match.iloc[0]['GEOID10']
if target_geoid not in df_comb['GEOID10'].values:
    st.error(ERROR_MSG); st.stop()

idx_row = df_comb[df_comb['GEOID10'] == target_geoid].index[0]
raw_scores = df_comb.iloc[idx_row][['s_e', 's_i', 's_h', 's_s']]

total_weight_sum = sum(weights_list)
actual_score = 4 * ( (raw_scores['s_e'] * w_e) + (raw_scores['s_i'] * w_i) + (raw_scores['s_h'] * w_h) + (raw_scores['s_s'] * w_s) ) / total_weight_sum

# Monte Carlo Simulation
x_matrix = df_comb[['s_e', 's_i', 's_h', 's_s']].to_numpy()
target_ratios = np.array(weights_list) / total_weight_sum
sim_weights = np.random.dirichlet(target_ratios * 30, 10000) 
sim_results = np.dot(sim_weights, x_matrix.T) * 4
local_sims = sim_results[:, idx_row]
m_loc, s_loc = norm.fit(local_sims)

if actual_score < 0.8: tier, color = "LOW IMPACT", "#2ecc71"
elif 0.8 <= actual_score < 1.6: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 1.6 <= actual_score < 2.4: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "EXTREME IMPACT (DANGER ZONE)", "#e74c3c"

st.header("📊 Impact Score Approximation")

st.markdown(f"""
<div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center;">
    <p style="color:white; font-size:1.2rem; margin:0; font-weight:bold; text-transform:uppercase;">{tier}</p>
    <h1 style="color:white; font-size:5rem; margin:0;">{actual_score:.2f} <span style="font-size:1.5rem;">± {s_loc:.3f}</span></h1>
    <p style="color:white; font-weight:bold;">Calculated Benefit for ZIP {zip_in}</p>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns([2, 1])
with col_l:
    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    ax_sim.hist(local_sims, bins=50, color='#aed6f1', density=True, alpha=0.7)
    x_range = np.linspace(min(local_sims), max(local_sims), 100)
    ax_sim.plot(x_range, norm.pdf(x_range, m_loc, s_loc), color='#2e86c1', lw=3)
    ax_sim.axvline(actual_score, color='#1b4f72', lw=3, label=f'Mean Impact: {actual_score:.2f}')
    ax_sim.set_title(f"Volatility Simulation for {zip_in} (10,000 Iterations)")
    ax_sim.set_xlabel("Impact Score"); ax_sim.legend(fontsize='x-small')
    st.pyplot(fig_sim)

with col_r:
    st.subheader("Statistical Interpretation")
    st.markdown(f"""
    This analysis measures the stability of the impact score by randomly shifting indicator priorities.
    * **Confidence Interval:** Narrower bounds (dotted red lines) indicate high data precision for this location[cite: 169].
    * **Robustness:** A low Standard Deviation (Volatility) confirms the score is stable across different weighting scenarios[cite: 169, 385].
    """)
    st.table(pd.DataFrame({"Metric": ["Calculated Score", "Volatility (SD)"], "Value": [f"{actual_score:.3f}", f"{s_loc:.3f}"]}))

# ----------------------------
# 3. COUNTY CONTEXT (CDF)
# ----------------------------
st.divider()
st.header("🌎 County-Wide Impact Ranking")
st.markdown("""
Every Census Tract in LA County is ranked from **Lowest Need (Left)** to **Highest Need (Right)**. 
Areas to the far right are priority "Danger Zones" where GKG projects offer maximum community value[cite: 160].
""")

medians = np.median(sim_results, axis=0)
sort_idx = np.argsort(medians)
fig_cdf, ax_cdf = plt.subplots(figsize=(12, 5))
ax_cdf.plot(medians[sort_idx], color='#1f77b4', lw=2.5, label='County Median Curve')
ax_cdf.scatter(np.searchsorted(medians[sort_idx], actual_score), actual_score, color='red', s=200, zorder=10, label=f'ZIP {zip_in} Rank')
ax_cdf.set_ylabel("Impact Score (0-4)"); ax_cdf.legend()
st.pyplot(fig_cdf)

# ----------------------------
# 4. PILLAR DEEP DIVE
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, weight, is_high_danger, source, anchor, calc):
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    sub = df[df['GEOID10'] == target_geoid]
    if sub.empty: st.warning(f"⚠️ Data missing for {name}."); return

    val = sub[col].values[0]
    std_val = raw_scores[score_key]
    mean_v, std_v = df[col].mean(), df[col].std()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.markdown(f"**Data Source:** {source}")
        st.markdown(f"**Methodology:** {calc}")
        st.metric(f"Raw Value", f"{val:,.1f} {unit}")
        st.metric("Pillar Contribution", f"{std_val:.3f} / 1.0")

        thresh = mean_v + std_v if is_high_danger else mean_v - std_v
        if (is_high_danger and val > thresh) or (not is_high_danger and val < thresh):
            st.error(f"🚨 **DANGER ZONE:** Exceeds critical threshold of {thresh:,.2f} {unit}[cite: 160].")
        else:
            st.success("✅ **NORMAL RANGE:** Within acceptable county bounds.")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(df[col].dropna(), bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(thresh, color='red', ls=':', lw=2, label='Danger Boundary (1 SD)')
        ax.set_xlabel(unit); ax.legend(fontsize='xx-small')
        st.pyplot(fig)
    st.divider()

pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Points', "", 's_e', 20, w_e, False, 
     "USC / Occidental College (2022)", "ejsm", "Standardized from 4-20 scale based on hazards, health, and canopy[cite: 437]."),
    (df_income, 'med_hh_income', 'Median Household Income', '$USD', "", 's_i', 250, w_i, False, 
     "US Census Bureau ACS (2021)", "income", "Inverted scale: lower income = higher score (Mean: $93,525)[cite: 468]."),
    (df_heat, 'DegHourDay', 'Heat Burden', 'Degree Hours/Day', "", 's_h', 150, w_h, True, 
     "Safe Clean Water Program LA (2022)", "heat", "Measures duration/intensity above 80°F (Median: 42.36)[cite: 376]."),
    (df_snap, 'SNAP_pct', 'Food Access (SNAP)', '% Pop', "", 's_s', 150, w_s, True, 
     "USDA Research Atlas (2019)", "snap", "Standardized percentage of population participating in SNAP[cite: 473].")
]

for p in sorted(pillars, key=lambda x: raw_scores[x[5]], reverse=True):
    plot_pillar(*p)

# ----------------------------
# 5. ARCGIS COUNTY NEED MAPPING
# ----------------------------
st.header("🗺️ ArcGIS County Need Mapping")
st.markdown("""
These spatial layers illustrate the geographic distribution of need across LA County. 
The maps help contextualize the Impact Score by showing where environmental justice and socioeconomic stressors intersect[cite: 177, 189].
""")

col_m1, col_m2 = st.columns(2)
# (Image loading logic remains same as provided in original code)

