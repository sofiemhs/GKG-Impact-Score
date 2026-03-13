import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- Weigh the data here!! ---
Enviro = 0.0
Income = 1.0
Heat = 1.0
Food = 1.0

# --- SECTION 0: GLOBAL WEIGHT CONFIGURATION ---
st.sidebar.header("⚖️ Impact Weighting")
st.sidebar.markdown("Adjust the importance of each pillar. Set to 0 to exclude a factor.")
w_e = st.sidebar.number_input("Environmental Justice Weight", min_value=0.0, value=Enviro, step=0.1)
w_i = st.sidebar.number_input("Income Weight", min_value=0.0, value=Income, step=0.1)
w_h = st.sidebar.number_input("Heat Burden Weight", min_value=0.0, value=Heat, step=0.1)
w_s = st.sidebar.number_input("Food Access Weight", min_value=0.0, value=Food, step=0.1)

weights_list = [w_e, w_i, w_h, w_s]

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Impact Score Analysis")

with st.expander("📖 Methodology, Data Sources & Years"):
    st.markdown(fr"""
    ### **The Question We Are Answering**
    "What impact does Good Karma Gardens' work have when converting spaces into gardens based on their location?"

    ### **Why Each Pillar Matters**
    * **[Environmental Justice (EJSM)](#environmental-justice-ejsm):** Communities with low Environmental Justice scores have a higher need for increased green space.
    * **[Median Household Income](#median-household-income):** Low-income communities are often overlooked and underserved regarding publicly accessible green spaces.
    * **[Heat Burden](#heat-burden):** Urban heat is a result of a lack of canopy cover and can cause dangerously high temperatures. Gardens limit this effect by actively cooling areas through transpiration.
    * **[Food Access (SNAP)](#food-access-snap):** Pinpoints 'food deserts' where affordable, fresh produce is scarce. Good Karma Gardens' work helps to alleviate this burden.

    ### **Standardization Logic**
    Every raw data point (dollars, degrees, or percentages) is standardized on a scale of **0.0 to 1.0**. 
    * **0.0** represents the lowest need/impact in the county.
    * **1.0** represents the highest need/impact in the county.

    ### **Impact Score Equation**
    The total **Impact Score (0.0 - 4.0)** is calculated as a weighted average of the four pillars, scaled to a maximum of 4. This ensures that even if one pillar is prioritized, the final score remains comparable across the county:
    
    $$Impact Score = 4 \times \frac{{\sum (w_{{i}} \times s_{{i}})}}{{\sum w_{{i}}}}$$
    
    **Current Active Weights:**
    * **Environmental Justice ($w_e$):** {w_e}
    * **Median Household Income ($w_i$):** {w_i}
    * **Heat Burden ($w_h$):** {w_h}
    * **Food Access ($w_s$):** {w_s}

    ### **Impact Ranges & Severity Logic**
    The following ranges represent the total potential need of a community. Because these four pillars often overlap, we assume that as the score increases, the community is experiencing **Systemic Compounding**—where multiple environmental and social stressors intersect to create a significantly higher state of vulnerability than a single factor alone.

    - <span style="color:#2ecc71; font-weight:bold;">0.0 - 0.8 (Low Impact):</span> Assumes 0% to 20% of total potential need.
    - <span style="color:#f1c40f; font-weight:bold;">0.8 - 1.6 (Medium Impact):</span> Assumes 20% to 40% of total potential need.
    - <span style="color:#e67e22; font-weight:bold;">1.6 - 2.4 (High Impact):</span> Assumes 40% to 60% of total potential need.
    - <span style="color:#e74c3c; font-weight:bold;">2.4 - 4.0 (Extreme Impact):</span> Assumes 60% to 100% of total potential need.
    """, unsafe_allow_html=True)

# --- NEW SECTION: APPENDIX & GLOSSARY ---
with st.expander("📚 Appendix: Glossary of Terms & Definitions"):
    st.markdown("""
    ### **Dashboard Specific Terms**
    * **Impact Score (0.0 - 4.0):** A composite number representing the 'need' of a specific area. A higher score indicates a higher priority for Good Karma Gardens' intervention due to intersecting social and environmental stressors.
    * **Standardization (0.0 - 1.0):** A mathematical process that converts different units (like dollars, percentages, and degrees) into a common scale. This allows us to compare "apples to oranges" (e.g., comparing income to heat levels).
    * **Systemic Compounding:** A term used here to describe how multiple disadvantages (like low income + high heat + low food access) don't just add up; they multiply the hardship a community faces.
    
    ### **Technical & Statistical Terms**
    * **Census Tract (GEOID):** A small, relatively permanent statistical subdivision of a county used by the US Census. It is more precise than a ZIP code for measuring local impact.
    * **Monte Carlo Simulation:** A computerized mathematical technique that allows people to account for risk and uncertainty. We use 10,000 "simulated" weight changes to see how stable the Impact Score is.
    * **Dirichlet Distribution:** A type of probability distribution used in our simulation to ensure that while we shift weights around, they always add up to a logical total.
    * **Standard Deviation (Volatility):** A measure of how much the score changes when you change the priorities (weights). A low volatility means the area has a high need regardless of which pillar you value most.
    
    ### **Environmental & Social Metrics**
    * **EJSM (Environmental Justice Screening Method):** A scoring system that identifies areas where residents are disproportionately burdened by pollution and social vulnerabilities.
    * **Degree Hours per Day:** A measurement of heat intensity over time. It doesn't just look at the high temperature, but how long the temperature stays dangerously high during a 24-hour period.
    * **Transpiration:** The process where plants release water vapor into the air. This acts like "natural air conditioning," which is why gardens are vital for lowering the **Heat Burden**.
    * **SNAP (Supplemental Nutrition Assistance Program):** Formerly known as Food Stamps. High participation in a tract often indicates a "food desert" where residents struggle to afford or access fresh produce.
    """)

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

    # 4. Food Access
    df_snap = pd.read_csv(f"{data_path}/Food_Deserts_CLEAN.csv")
    df_snap.columns = df_snap.columns.str.strip()
    
    def format_geoid(x):
        s = str(x).split('.')[0].strip()
        if len(s) <= 7: # It's a 6-digit tract; add CA (06) + LA (037)
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

ERROR_MSG = "The inputted zip code either doesn't exist within Los Angeles County or doesn't have any reliable data reported. Please try another Zipcode."

if match.empty:
    st.error(ERROR_MSG)
    st.stop()

target_geoid = match.iloc[0]['GEOID10']
if target_geoid not in df_comb['GEOID10'].values:
    st.error(ERROR_MSG)
    st.stop()

idx_row = df_comb[df_comb['GEOID10'] == target_geoid].index[0]
raw_scores = df_comb.iloc[idx_row][['s_e', 's_i', 's_h', 's_s']]

missing_info_count = (raw_scores == 0).sum()
if missing_info_count >= 3:
    st.error(ERROR_MSG)
    st.stop()

# --- CUSTOM WEIGHING CALCULATION ---
total_weight_sum = sum(weights_list)

if total_weight_sum == 0:
    actual_score = 0.0
    st.sidebar.warning("Total weight is 0. Impact score cannot be calculated.")
else:
    actual_score = 4 * ( (raw_scores['s_e'] * w_e) + (raw_scores['s_i'] * w_i) + (raw_scores['s_h'] * w_h) + (raw_scores['s_s'] * w_s) ) / total_weight_sum

# Monte Carlo: 10,000 simulations
x_matrix = df_comb[['s_e', 's_i', 's_h', 's_s']].to_numpy()
if total_weight_sum > 0:
    target_ratios = np.array(weights_list) / total_weight_sum
    # Dirichlet needs strictly positive alpha. We add a tiny epsilon to 0-weight items to allow simulation.
    alpha_vals = np.maximum(target_ratios * 30, 0.0001)
    sim_weights = np.random.dirichlet(alpha_vals, 10000) 
    sim_results = np.dot(sim_weights, x_matrix.T) * 4
    local_sims = sim_results[:, idx_row]
    m_loc, s_loc = norm.fit(local_sims)
else:
    sim_results = np.zeros((10000, len(df_comb)))
    local_sims = np.zeros(10000)
    m_loc, s_loc = 0, 0

# Impact Range Logic
if actual_score < 0.8: tier, color = "LOW IMPACT", "#2ecc71"
elif 0.8 <= actual_score < 1.6: tier, color = "MEDIUM IMPACT", "#f1c40f"
elif 1.6 <= actual_score < 2.4: tier, color = "HIGH IMPACT", "#e67e22"
else: tier, color = "EXTREME IMPACT (DANGER ZONE)", "#e74c3c"

st.header("📊 Impact Score Approximation")

st.markdown(f"""
<div style="background-color:{color}; padding:20px; border-radius:15px; border: 2px solid rgba(0,0,0,0.1); margin-bottom:20px; text-align:center;">
    <p style="color:white; font-size:1.2rem; margin:0; font-weight:bold; text-transform:uppercase;">{tier}</p>
    <h1 style="color:white; font-size:5rem; margin:0; line-height:1;">{actual_score:.2f} <span style="font-size:1.5rem;">± {s_loc:.3f}</span></h1>
    <p style="color:white; font-weight:bold; margin-top:10px;">Calculated Impact Score for ZIP {zip_in}</p>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns([2, 1])
with col_l:
    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    if total_weight_sum > 0:
        ax_sim.hist(local_sims, bins=50, color='#aed6f1', density=True, alpha=0.7)
        x_range = np.linspace(min(local_sims), max(local_sims), 100)
        ax_sim.plot(x_range, norm.pdf(x_range, m_loc, s_loc), color='#2e86c1', lw=3)
        ax_sim.axvline(actual_score, color='#1b4f72', lw=3, label=f'Impact Score (Mean): {actual_score:.2f}')
        ax_sim.axvline(actual_score - s_loc, color='#e74c3c', ls=':', lw=2, label='Confidence Bounds')
        ax_sim.axvline(actual_score + s_loc, color='#e74c3c', ls=':', lw=2)
    else:
        ax_sim.text(0.5, 0.5, "No Weights Active", ha='center', va='center')
    
    ax_sim.set_title(f"Score Variance Simulation for {zip_in} (10,000 Iterations)")
    ax_sim.set_xlabel("Impact Score")
    ax_sim.set_ylabel("Probability Density")
    ax_sim.legend(fontsize='x-small')
    st.pyplot(fig_sim)

with col_r:
    st.subheader("Statistical Interpretation")
    st.markdown(f"""
    This graph analyzes the stability of the impact score using **10,000 simulations**. By randomly shifting the priority of our four pillars, we determine the certainty of the final score. 
    
    - **Blue Line:** The theoretical distribution of scores.
    - **Dotted Red Lines:** Represent the Confidence Interval. **The closer these lines are to the mean, the more certain we are of the data's precision in this location.**
    - **Standard Deviation:** A measure of "spread." A small number indicates the score remains consistent across different weighting scenarios.
    """)
    st.table(pd.DataFrame({
        "Metric": ["Calculated Score", "Standard Deviation (Volatility)"],
        "Value": [f"{actual_score:.3f}", f"{s_loc:.3f}"]
    }))

# ----------------------------
# 3. COUNTY CONTEXT (CDF)
# ----------------------------
st.divider()
st.header("🌎 County-Wide Impact Ranking")

st.markdown("""
This plot ranks every Census Tract in Los Angeles County from **Lowest Need (Left)** to **Highest Need (Right)**.
The blue shaded area represents the 'Average' middle 50% of the county. Areas to the far right are priority zones for Good Karma Gardens.
""")

medians = np.median(sim_results, axis=0)
p25 = np.percentile(sim_results, 25, axis=0)
p75 = np.percentile(sim_results, 75, axis=0)
sort_idx = np.argsort(medians)

fig_cdf, ax_cdf = plt.subplots(figsize=(12, 5))
ax_cdf.plot(medians[sort_idx], color='#1f77b4', lw=2.5, label='LA County Median Curve')
ax_cdf.fill_between(range(len(medians)), p25[sort_idx], p75[sort_idx], color='#1f77b4', alpha=0.2, label='25th-75th Percentile')

rank_pos = np.searchsorted(medians[sort_idx], actual_score)
ax_cdf.scatter(rank_pos, actual_score, color='red', s=200, zorder=10, label=f'ZIP {zip_in} Rank', edgecolor='white')

ax_cdf.grid(True, linestyle='-', alpha=0.2)
ax_cdf.set_ylabel("Impact Score (0-4)")
ax_cdf.set_xlabel("Census Tracts (Sorted by Need)")
ax_cdf.legend(loc='lower right')
st.pyplot(fig_cdf)

percentile = (medians < actual_score).mean() * 100
if percentile > 75:
    st.warning(f"📍 **ZIP {zip_in}** is in the top **{100-percentile:.1f}%** of high-need areas in the county. Its need is significantly higher than most of LA.")
elif percentile < 25:
    st.success(f"📍 **ZIP {zip_in}** is in the bottom **{percentile:.1f}%** of need areas. Its need is lower than most of the county.")
else:
    st.info(f"📍 **ZIP {zip_in}** is in the middle range, with a higher need than **{percentile:.1f}%** of LA tracts.")

# ----------------------------
# 4. PILLAR DEEP-DIVE
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep-Dive")

def plot_pillar(df, col, name, unit, desc, score_key, bins, weight, is_high_danger=True, source="", anchor_id="", source_url=""):
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
    sub = df[df['GEOID10'] == target_geoid]
    
    if sub.empty:
        st.warning(f"⚠️ **DATA MISSING:** No local reporting for **{name}**. Standardized Score = **0.0**.")
        st.divider(); return

    val = sub[col].values[0]
    std_val = raw_scores[score_key]
    
    # Calculate percentage contribution to the total impact score
    weighted_contrib = std_val * weight
    total_weighted_points = sum([raw_scores['s_e']*w_e, raw_scores['s_i']*w_i, raw_scores['s_h']*w_h, raw_scores['s_s']*w_s])
    pct_contrib = (weighted_contrib / total_weighted_points * 100) if total_weighted_points > 0 else 0

    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.markdown(f"**Description:** {desc}")
        st.markdown(f"**Cite source:** [{source}]({source_url})")
        
        if weight == 0:
            st.info("🚫 **EXCLUDED:** This pillar's weight is set to 0 and is not currently impacting the total score.")
        
        st.metric(f"ZIP {zip_in} Raw Value", f"{val:,.1f} {unit}")
        st.metric("Pillar Score Contribution", f"{std_val:.3f} / 1.0")
        st.metric("% of Total Impact Score", f"{pct_contrib:.1f}%")
        
        thresh = mean_v + std_v if is_high_danger else mean_v - std_v
        if (is_high_danger and val > thresh) or (not is_high_danger and val < thresh):
            st.error("🚨 **DANGER ZONE:** This metric exceeds the critical threshold.")
        else:
            st.success("✅ **NORMAL RANGE:** This metric is within acceptable bounds.")

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
        ax.axvline(mean_v + std_v, color='red', ls=':', lw=2, label='±1 SD')
        ax.axvline(mean_v - std_v, color='red', ls=':', lw=2)
        
        ax.set_xlabel(f"{unit}")
        ax.set_ylabel("Frequency Density")
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)
    st.divider()

pillars = [
    (df_ejsm, 'CIscore', 'Environmental Justice (EJSM)', 'Points', 
     "This metric evaluates environmental justice across hazard proximity, health risk, social vulnerability, and canopy cover. Scores range from 4 to 20. Areas scoring more than 1 SD below the mean (< 7.65) are identified as having the highest need for community green space.", 
     's_e', 20, w_e, False, "USC / Occidental College / LA County (2022)", "environmental-justice-ejsm", "https://data.lacounty.gov/apps/lacounty::green-zones-program-ejsm-2/about?path="),
    
    (df_income, 'med_hh_income', 'Median Household Income', '$USD', 
     "Measures the median income for households within the tract. With a county-wide mean income of $93,525, 'Danger Zones' are defined as tracts at or below $53,423. Lower income levels directly correlate to fewer private green spaces and higher climate vulnerability.", 
     's_i', 250, w_i, False, "US Census Bureau ACS 5-Year Estimates (2021)", "median-household-income", "https://data.lacounty.gov/datasets/lacounty::median-household-income/about"),
    
    (df_heat, 'DegHourDay', 'Heat Burden', 'Degree Hours per Day', 
     "Measured in 'Degree Hours per Day,' which tracks how many degrees—and for how long—the local temperature exceeds a baseline of 80°F. With a county median of 42.36, areas exceeding 82.61 are 'Danger Zones.' High numbers indicate intense, sustained heat exposure that can be mitigated by garden transpiration.", 
     's_h', 150, w_h, True, "Safe Clean Water Program LA (2022)", "heat-burden", "https://scwp-lacounty.hub.arcgis.com/datasets/lacounty::urban-heat-island-index/about"),
    
    (df_snap, 'SNAP_pct', 'Food Access (SNAP)', '% Pop', 
     "Calculated as (SNAP Participants / Total Population) * 100. The county mean is 3.15%. 'Danger Zones' exceed 5.52% participation, pinpointing 'food deserts' where gardens alleviate the burden of fresh produce.", 
     's_s', 150, w_s, True, "USDA Food Access Research Atlas (2019)", "food-access-snap", "https://geohub.lacity.org/datasets/lacounty::food-deserts/about")
]

for p in sorted(pillars, key=lambda x: raw_scores[x[5]], reverse=True):
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], source=p[9], anchor_id=p[10], source_url=p[11])


# ----------------------------
# 5. ARCGIS COUNTY NEED MAPPING
# ----------------------------
st.header("🗺️ ArcGIS County Need Mapping")
st.markdown("""
The following spatial data layers illustrate the geographic distribution of need across Los Angeles County. 
These visuals help contextualize the **Impact Score** by showing where environmental and social stressors intersect on a regional scale.
""")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("⚖️ Environmental Justice (EJSM)")
    map_ejsm = "GKG-Impact-Score/map_photos/EJSM.png"
    if os.path.exists(map_ejsm):
        st.image(map_ejsm, use_container_width=True)
    else:
        st.info("Map not found at specified path.")
    st.caption("Visualizes cumulative hazard proximity and health risks.")

    st.subheader("🥦 Food Access (SNAP)")
    map_food = "GKG-Impact-Score/map_photos/food_access.png"
    if os.path.exists(map_food):
        st.image(map_food, use_container_width=True)
    else:
        st.info("Map not found at specified path.")
    st.caption("Highlights 'food deserts' and SNAP participation density.")

with col_m2:
    st.subheader("🔥 Urban Heat Burden")
    map_heat = "GKG-Impact-Score/map_photos/urban_heat.png"
    if os.path.exists(map_heat):
        st.image(map_heat, use_container_width=True)
    else:
        st.info("Map not found at specified path.")
    st.caption("Displays areas with high degree-hour exposure and low canopy cover.")

    st.subheader("💰 Median Household Income")
    map_income = "GKG-Impact-Score/map_photos/median_household_income.png"
    if os.path.exists(map_income):
        st.image(map_income, use_container_width=True)
    else:
        st.info("Map not found at specified path.")
    st.caption("Identifies underserved low-income tracts across the county.")


