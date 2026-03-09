import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: MISSION & METHODOLOGY ---
st.title("🌿 Good Karma Gardens: Precision Impact Analysis")
st.markdown("""
### **Objective**
Good Karma Gardens utilizes this dashboard to identify Los Angeles County neighborhoods where **lawn-to-garden conversions** provide the highest return on investment for climate resilience, public health, and economic stability. By replacing water-intensive lawns with native, food-producing landscapes, we directly mitigate the specific vulnerabilities identified below.
""")

with st.expander("📖 Methodology: How We Measure Impact (0.0 - 4.0 Scale)"):
    st.markdown("""
    To compare disparate data types (dollars vs. temperature), we utilize **Min-Max Scaling**. Each factor is converted into a 'Pillar Score' ranging from **0.0 (Lowest Need)** to **1.0 (Highest Need)**.
    
    **The Four Pillars of Impact:**
    1. **Environmental Justice (EJSM):** A composite of pollution exposure (air/water) and existing health vulnerabilities.
    2. **Economic Need:** Median Household Income. Lower income tracts receive higher impact scores to prioritize resource equity.
    3. **Heat Burden:** Degree-Hour Days. Measures the intensity of the **Urban Heat Island** effect.
    4. **Food Access:** Percentage of households utilizing SNAP. Acts as a proxy for food insecurity and 'food desert' conditions.

    **The Equation:**
    """)
    st.latex(r"Impact Score = EJSM_{std} + Economic_{std} + Heat_{std} + Food_{std}")
    st.info("The final score ranges from **0.0 to 4.0**. A higher score represents a greater intersection of environmental and social need.")

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

    # Heat
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

    # Calculate Zip-Level Means for the ranking table
    df_zip_rankings = df_ziptract.merge(df_comb, on='GEOID10')
    df_zip_rankings['Total_Score'] = df_zip_rankings[['s_e','s_i','s_h','s_s']].sum(axis=1)
    zip_summary = df_zip_rankings.groupby('ZIP')['Total_Score'].mean().reset_index()
    zip_summary.columns = ['ZIP Code', 'Mean Impact Score']
    zip_summary = zip_summary.sort_values(by='Mean Impact Score', ascending=False)

    return df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb, zip_summary

try:
    df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb, zip_summary = load_all_data()
except Exception as e:
    st.error(f"❌ Error Loading Data: {e}")
    st.stop()

# ----------------------------
# 2. LOCAL ANALYSIS
# ----------------------------

st.sidebar.header("📍 Location Input")
zip_in = st.sidebar.text_input("Enter LA County ZIP:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if match.empty:
    st.error(f"❌ ZIP {zip_in} is outside LA County or missing from dataset.")
    st.stop()

target_geoid = match.iloc[0]['GEOID10']
idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]

if len(idx) > 0:
    raw_pillar_scores = df_comb.iloc[idx[0]][['s_e','s_i','s_h','s_s']]
    actual_score = raw_pillar_scores.sum()
    
    # Monte Carlo Sensitivity (Scaled 0-4)
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    weights = np.random.uniform(0, 1, (10000, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T) * 4 
    
    d = sim_results[:, idx[0]]
    m, s = norm.fit(d)
    
    # Tweak 3: Reformatted Driver Text
    labels = {'s_e': 'EJSM', 's_i': 'Economic Need', 's_h': 'Heat', 's_s': 'Food Access'}
    drivers = [labels[k] for k, v in (raw_pillar_scores / actual_score).items() if v >= 0.30] if actual_score > 0 else []
    driver_text = f" Score is driven by **{', '.join(drivers)}**" if drivers else ""

    if actual_score < 0.8: tier, color, desc = "LOW IMPACT", "#2ecc71", "Healthy baseline metrics."
    elif 0.8 <= actual_score < 1.6: tier, color, desc = "MEDIUM IMPACT", "#f1c40f", "Emerging needs detected."
    elif 1.6 <= actual_score < 2.4: tier, color, desc = "HIGH IMPACT", "#e67e22", "Significant vulnerability."
    else: tier, color, desc = "VERY HIGH IMPACT", "#e74c3c", "CRITICAL PRIORITY: Critical priority."

    st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
        <p style="color:white; font-size:1.4rem; margin-top:5px; font-weight:bold;">{desc}{driver_text}</p></div>""", unsafe_allow_html=True)

    st.header(f"📊 Impact Sensitivity Analysis: ZIP {zip_in}")
    st.write("This simulation tests the score's stability by shifting the importance (weights) of the four pillars across 10,000 scenarios.")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(d, bins=30, color='#aed6f1', density=True, alpha=0.7)
        x_range = np.linspace(min(d), max(d), 100)
        ax.plot(x_range, norm.pdf(x_range, m, s), color='#2e86c1', lw=3, label='Simulation Fit')
        ax.axvline(actual_score, color='#1b4f72', lw=3, label=f'Actual Sum: {actual_score:.2f}')
        ax.axvline(actual_score-s, color='#e74c3c', ls='--', label='Lower Confidence Bounds')
        ax.axvline(actual_score+s, color='#e74c3c', ls='--', label='Upper Confidence Bounds')
        ax.set_xlabel("Impact Score (0.0 - 4.0)")
        ax.legend(fontsize='x-small')
        st.pyplot(fig)
    with col_r:
        st.metric("Total Impact Score", f"{actual_score:.3f} / 4.0")
        st.write("**Simulation Stats**")
        st.table(pd.DataFrame({
            "Metric": ["Average Outcome", "Volatility (SD)", "68% Confidence Interval"],
            "Value": [f"{m:.3f}", f"{s:.3f}", f"{actual_score-s:.2f} — {actual_score+s:.2f}"]
        }))

# --- SECTION 3: COUNTY CONTEXT & RANKINGS ---
st.divider()
st.header("🌎 LA County Contextual Rankings")

col_cdf, col_table = st.columns([2, 1])

with col_cdf:
    st.subheader("Cumulative Need Distribution")
    st.write("The shaded area represents the 'Middle 50%' of LA County (25th to 75th percentile).")
    
    all_tract_sums = x_matrix.sum(axis=1)
    sorted_sums = np.sort(all_tract_sums)
    n = len(sorted_sums)
    p25_idx, p75_idx = int(0.25 * n), int(0.75 * n)

    fig_cdf, ax_cdf = plt.subplots(figsize=(10, 5))
    ax_cdf.plot(sorted_sums, color='#2980b9', lw=3)
    
    # Tweak 4: Highlight 25th-75th percentile
    ax_cdf.fill_between(range(p25_idx, p75_idx), sorted_sums[p25_idx:p75_idx], 
                        color='#f1c40f', alpha=0.4, label='Interquartile Range (25th-75th)')
    
    # Red dot for current ZIP
    rank_idx = np.searchsorted(sorted_sums, actual_score)
    ax_cdf.scatter(rank_idx, actual_score, color='red', s=120, zorder=5, label=f'ZIP {zip_in} Rank')
    
    ax_cdf.set_ylabel("Impact Score (0-4)")
    ax_cdf.set_xlabel("Census Tracts (Ranked by Need)")
    ax_cdf.legend()
    st.pyplot(fig_cdf)

with col_table:
    st.subheader("Mean Score by ZIP")
    st.write("Average impact score across all census tracts within each ZIP code.")
    st.dataframe(zip_summary, height=400, hide_index=True)

# --- SECTION 4: THE FOUR PILLARS ---
st.divider()
st.header("🔍 Pillar Deep-Dive")
st.write("Detailed distribution analysis for each of the four data pillars used in the calculation.")



def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True):
    subset = df[df['GEOID10'] == target_geoid]
    if subset.empty:
        st.warning(f"⚠️ Local data for {name} is unavailable for this tract.")
        return

    local_val = subset[col].values[0]
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    weight = df_comb.iloc[idx[0]][score_key]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.markdown(f"*{desc}*")
        st.metric(f"ZIP {zip_in}", f"{local_val:,.1f} {unit}")
        st.write(f"**Pillar Weight:** {weight:.3f} points")
    with c2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        counts, bin_edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        for i in range(len(patches)):
            mid = (bin_edges[i] + bin_edges[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        x = np.linspace(data.min(), data.max(), 500)
        ax.plot(x, norm.pdf(x, mean_v, std_v), color='black', lw=2)
        ax.axvline(local_val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(mean_v, color='black', lw=1.5, label='LA Mean')
        ax.axvline(mean_v - std_v, color='gray', ls=':', label='-1 SD')
        ax.axvline(mean_v + std_v, color='gray', ls=':', label='+1 SD')
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)



plot_pillar(df_ejsm, 'CIscore', 'Environmental Justice', 'Points', "Measures cumulative pollution burden and population vulnerability.", 's_e', 20, False)
st.divider()
plot_pillar(df_income, 'med_hh_income', 'Median HH Income', '$USD', "Indicator of economic resilience and funding priority.", 's_i', 250, False)
st.divider()
plot_pillar(df_heat, 'DegHourDay', 'Heat Burden', 'Days', "Degree-hour days above baseline; identifies critical heat-island mitigation zones.", 's_h', 150, True)
st.divider()
plot_pillar(df_snap, 'SNAP_pct', 'Food Access', '% Pop', "Percentage of households receiving SNAP; proxy for food sovereignty needs.", 's_s', 150, True)
