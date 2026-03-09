import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: MISSION & PILLAR LOGIC ---
# (1) Title Change
st.title("🌿 Good Karma Gardens: Impact Score Analysis")

# (2) Detailed Methodology Dropdown
with st.expander("📖 Methodology, Data Sources & Years"):
    st.markdown("""
    ### **The Question We Are Answering**
    *"Where in Los Angeles County can a lawn-to-garden conversion provide the most significant uplift for climate resilience, public health, and economic equity?"*

    ### **Why Each Pillar Matters**
    1. **Environmental Justice (EJSM):** Identifies areas where residents face cumulative burdens from pollution (air, water, toxic waste) combined with biological vulnerabilities.
    2. **Economic Need:** Measures the financial capacity of a neighborhood. Lower-income areas often lack the resources for private greening, making community-led gardens a critical resource.
    3. **Heat Burden:** Targets 'Urban Heat Islands' where lack of canopy cover causes dangerously high temperatures. Gardens actively cool these areas through transpiration.
    4. **Food Access (SNAP):** Pinpoints 'food deserts' where affordable, fresh produce is scarce. Gardens here serve as decentralized grocery stores.

    ### **Standardization Logic**
    Every raw data point (dollars, degrees, or percentages) is standardized on a scale of **0.0 to 1.0**. 
    * **0.0** represents the lowest need/impact in the county.
    * **1.0** represents the highest need/impact in the county.
    * The total **Impact Score (0.0 - 4.0)** is the sum of these four pillars.

    ### **Impact Ranges**
    - **0.0 - 0.8 (Low Impact):** Healthy baseline; baseline resilience present.
    - **0.8 - 1.6 (Medium Impact):** Emerging needs; localized vulnerabilities detected.
    - **1.6 - 2.4 (High Impact):** Significant need; multi-factor vulnerabilities present.
    - **2.4 - 4.0 (Extreme Impact):** **Danger Zone**; critical intersection of pollution, poverty, and climate risk.
    """)
    
    st.markdown("""
    | Pillar | Source | Year | Range (Danger Threshold) |
    | :--- | :--- | :--- | :--- |
    | **Env. Justice** | SCAG / CalEnviroScreen | 2022 | > 1 Standard Deviation from Mean |
    | **Economic Need** | ACS 5-Year Estimates | 2021 | Inverted (Lower income = Higher Score) |
    | **Heat Burden** | NOAA / Urban Heat Watch | 2022 | > 1 Standard Deviation from Mean |
    | **Food Access** | USDA Food Research Atlas | 2019 | > 1 Standard Deviation from Mean |
    """)

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

# Monte Carlo (Local)
x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
sim_weights = np.random.uniform(0, 1, (1000, 4))
sim_weights /= sim_weights.sum(axis=1, keepdims=True)
sim_results = np.dot(sim_weights, x_matrix.T) 
local_sims = sim_results[:, idx_row] * 4
m_loc, s_loc = norm.fit(local_sims)

# (3) Change Title
st.header("📊 Impact Score Approximation")

# (4) Clearly and largely display the Impact Score
st.markdown("### **Calculated Impact Score**")
st.markdown(f"""
<div style="background-color:#f8f9fa; padding:10px; border-radius:10px; border-left: 10px solid #2e86c1; margin-bottom:20px;">
    <h1 style="color:#1b4f72; font-size:4rem; margin:0;">{m_loc:.2f} <span style="font-size:1.5rem;">± {s_loc:.3f}</span></h1>
    <p style="color:#2e86c1; font-weight:bold; margin:0;">Estimated resilience ROI based on local sensitivity simulation.</p>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns([2, 1])
with col_l:
    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    ax_sim.hist(local_sims, bins=30, color='#aed6f1', density=True, alpha=0.7)
    x_range = np.linspace(min(local_sims), max(local_sims), 100)
    ax_sim.plot(x_range, norm.pdf(x_range, m_loc, s_loc), color='#2e86c1', lw=3)
    ax_sim.axvline(actual_score, color='#1b4f72', lw=3, label=f'Raw Calculation: {actual_score:.2f}')
    ax_sim.set_title(f"Score Variance Simulation for {zip_in}")
    ax_sim.legend(fontsize='x-small')
    st.pyplot(fig_sim)

with col_r:
    # (5) Explanation of Approximation
    st.subheader("What this shows:")
    st.markdown("""
    This histogram represents **1,000 simulations** of your local data. We vary the importance (weights) of the four pillars to see how stable your score is. 
    
    - **Simulation Mean:** The most likely impact score for this area.
    - **Volatility (SD):** Shows how much the score changes if we prioritize one pillar over another.
    - **Bounds:** The range where the score falls 68% of the time.
    """)
    st.table(pd.DataFrame({
        "Metric": ["Calculated Score", "Simulated Mean", "Standard Deviation", "Confidence Lower", "Confidence Upper"],
        "Value": [f"{actual_score:.3f}", f"{m_loc:.3f}", f"{s_loc:.3f}", f"{actual_score-s_loc:.2f}", f"{actual_score+s_loc:.2f}"]
    }))

# ----------------------------
# 3. COUNTY CONTEXT (FIXED CDF)
# ----------------------------
st.divider()
st.header("🌎 County-Wide Impact Ranking")

# (6) Explain Plot
st.markdown("""
This plot ranks every single Census Tract in Los Angeles County from **Lowest Need (Left)** to **Highest Need (Right)**.
The blue shaded area represents the 'Average' middle 50% of the county. Areas to the far right are the priority zones for Good Karma Gardens.
""")

medians = np.median(sim_results, axis=0)
p25 = np.percentile(sim_results, 25, axis=0)
p75 = np.percentile(sim_results, 75, axis=0)
sort_idx = np.argsort(medians)

fig_cdf, ax_cdf = plt.subplots(figsize=(12, 5))
ax_cdf.plot(medians[sort_idx], color='#1f77b4', lw=2.5, label='LA County Median Curve')
ax_cdf.fill_between(range(len(medians)), p25[sort_idx], p75[sort_idx], color='#1f77b4', alpha=0.2, label='25th-75th Percentile')

# (7) Add dot for local ZIP
rank_pos = np.searchsorted(medians[sort_idx], medians[idx_row])
ax_cdf.scatter(rank_pos, medians[idx_row], color='red', s=200, zorder=10, label=f'ZIP {zip_in} Rank', edgecolor='white')

ax_cdf.grid(True, linestyle='-', alpha=0.2)
ax_cdf.set_ylabel("Weighted Standardized Index")
ax_cdf.set_xlabel("Census Tracts (Sorted by Need)")
ax_cdf.legend(loc='lower right')
st.pyplot(fig_cdf)

# (8) Placement Message
percentile = (medians < medians[idx_row]).mean() * 100
if percentile > 75:
    st.warning(f"📍 **ZIP {zip_in}** is in the top **{100-percentile:.1f}%** of high-need areas in the county. It is significantly higher than most of LA.")
elif percentile < 25:
    st.success(f"📍 **ZIP {zip_in}** is in the bottom **{percentile:.1f}%** of need areas. It is lower than most of the county.")
else:
    st.info(f"📍 **ZIP {zip_in}** is in the middle range, higher than **{percentile:.1f}%** of LA tracts.")

# ----------------------------
# 4. PILLAR DEEP-DIVE
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep-Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True, source=""):
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
        # (11) Pillar Description and Source
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
        # (9) Histogram with Red Danger Zones
        counts, edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        thresh_line = mean_v + std_v if is_high_danger else mean_v - std_v
        for i in range(len(patches)):
            mid = (edges[i] + edges[i+1]) / 2
            if (is_high_danger and mid > thresh_line) or (not is_high_danger and mid < thresh_line):
                patches[i].set_facecolor('#e74c3c')
        
        # (10) SD Curve Fit
        x = np.linspace(data.min(), data.max(), 500)
        ax.plot(x, norm.pdf(x, mean_v, std_v), color='black', lw=2, label='Normal Distribution')
        
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(mean_v + std_v, color='red', ls=':', lw=2, label='+1 SD')
        ax.axvline(mean_v - std_v, color='red', ls=':', lw=2, label='-1 SD')
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)
    st.divider()

# Detailed Pillar Data for Loop
pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice', 'Points', "Combines pollution exposure with population sensitivity.", 's_e', 20, False, "SCAG / CalEnviroScreen 4.0 (2022)"),
    (df_income, 'med_hh_income', 'Median HH Income', '$USD', "Inverted to show higher impact for lower income communities.", 's_i', 250, False, "US Census Bureau ACS 5-Year Estimates (2021)"),
    (df_heat, 'DegHourDay', 'Heat Burden', 'Days', "Degree-hour days above the local baseline.", 's_h', 150, True, "NOAA Urban Heat Watch (2022)"),
    (df_snap, 'SNAP_pct', 'Food Access', '% Pop', "Households using SNAP; indicates risk of food insecurity.", 's_s', 150, True, "USDA Food Access Research Atlas (2019)")
]

for p in sorted(pillars, key=lambda x: raw_scores[x[5]], reverse=True):
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], source=p[8])
