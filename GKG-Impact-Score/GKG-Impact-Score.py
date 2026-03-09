import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: MISSION & PILLAR LOGIC ---
st.title("🌿 Good Karma Gardens: Precision Impact Analysis")

st.markdown("""
### **Objective**
Good Karma Gardens uses this tool to prioritize **lawn-to-garden conversions**. We don't just look for empty space; we look for the intersection of environmental burden and social need to ensure every garden we build provides maximum community resilience.
""")

with st.expander("📖 Methodology & The 'Why' Behind Our Data"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### **Why these four pillars?**")
        st.write("**1. Environmental Justice (EJSM):** Pollution isn't distributed evenly. We use this to find communities already burdened by poor air quality and toxic exposure.")
        st.write("**2. Economic Need:** Wealthy areas can afford private landscaping. We prioritize low-income tracts where green space is a luxury, not a given.")
    with col_b:
        st.write("**3. Heat Burden:** Urban 'Heat Islands' can be 10°F hotter than suburbs. Gardens cool the air. We target the hottest streets to save lives.")
        st.write("**4. Food Access:** In 'food deserts,' fresh produce is expensive or unavailable. Our gardens turn lawns into local grocery stores.")
    
    st.latex(r"Total Impact (0.0 - 4.0) = EJSM_{std} + Income_{std} + Heat_{std} + Food_{std}")
    st.warning("⚠️ **Assumption:** If data for a specific pillar is missing from the Census, it is recorded as **0** to ensure a baseline calculation is still possible.")

# ----------------------------
# 1. Data Loading & Danger Zone Calculations
# ----------------------------

@st.cache_data
def load_all_data():
    # [Data loading paths and logic remain consistent with previous versions]
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
    
    # Ranking Table
    df_zip_rankings = df_ziptract.merge(df_comb, on='GEOID10')
    df_zip_rankings['Total_Score'] = df_zip_rankings[['s_e','s_i','s_h','s_s']].sum(axis=1)
    zip_summary = df_zip_rankings.groupby('ZIP')['Total_Score'].mean().reset_index()
    zip_summary.columns = ['ZIP Code', 'Mean Impact Score']
    zip_summary = zip_summary.sort_values(by='Mean Impact Score', ascending=False)

    return df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb, zip_summary

df_ejsm, df_income, df_heat, df_snap, df_ziptract, df_comb, zip_summary = load_all_data()

# ----------------------------
# 2. LOCAL ANALYSIS & ERROR TABLE
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

# Monte Carlo (10k sims)
x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
weights = np.random.uniform(0, 1, (10000, 4))
weights /= weights.sum(axis=1, keepdims=True)
sim_results = np.dot(weights, x_matrix.T) * 4
d = sim_results[:, idx_row]
m, s_dist = norm.fit(d)

# Danger Zone Definition
is_danger = actual_score >= 2.4
total_danger_tracts = (df_comb[['s_e','s_i','s_h','s_s']].sum(axis=1) >= 2.4).sum()

# Status Banner
if actual_score < 0.8: tier, color, desc = "LOW IMPACT", "#2ecc71", "Healthy baseline metrics."
elif 0.8 <= actual_score < 1.6: tier, color, desc = "MEDIUM IMPACT", "#f1c40f", "Emerging needs detected."
elif 1.6 <= actual_score < 2.4: tier, color, desc = "HIGH IMPACT", "#e67e22", "Significant vulnerability."
else: tier, color, desc = "VERY HIGH IMPACT (DANGER ZONE)", "#e74c3c", "CRITICAL PRIORITY: Intervention required."

# Driver Text
labels = {'s_e': 'EJSM', 's_i': 'Economic Need', 's_h': 'Heat', 's_s': 'Food Access'}
drivers = [labels[k] for k, v in (raw_scores / actual_score).items() if v >= 0.30] if actual_score > 0 else []
driver_info = f" Score is driven by **{', '.join(drivers)}**" if drivers else ""

st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
    <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
    <p style="color:white; font-size:1.4rem; margin-top:5px; font-weight:bold;">{desc}{driver_info}</p></div>""", unsafe_allow_html=True)

# Danger Zone Callout
if is_danger:
    st.error(f"🚨 **ALERT:** ZIP {zip_in} is currently in the **Danger Zone**. This tract is among the {total_danger_tracts} most vulnerable tracts in LA County.")
else:
    st.success(f"✅ **NOTICE:** ZIP {zip_in} is **NOT** in the Danger Zone (Score < 2.4). While needs exist, it does not meet the critical intervention threshold.")

st.header(f"📊 Statistical Sensitivity & Error Analysis")
col_l, col_r = st.columns([2, 1])
with col_l:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(d, bins=30, color='#aed6f1', density=True, alpha=0.7)
    x_range = np.linspace(min(d), max(d), 100)
    ax.plot(x_range, norm.pdf(x_range, m, s_dist), color='#2e86c1', lw=3)
    ax.axvline(actual_score, color='#1b4f72', lw=3, label=f'Current ZIP: {actual_score:.2f}')
    ax.axvline(actual_score - s_dist, color='#e74c3c', ls='--', label='-1 SD (Margin of Error)')
    ax.axvline(actual_score + s_dist, color='#e74c3c', ls='--', label='+1 SD (Margin of Error)')
    ax.set_title(f"Score Variance Simulation for {zip_in}")
    ax.legend(fontsize='x-small')
    st.pyplot(fig)

with col_r:
    st.subheader("Statistical Breakdown")
    st.table(pd.DataFrame({
        "Metric": ["Calculated Impact", "Simulation Mean", "Variance (SD)", "Lower Bound", "Upper Bound"],
        "Value": [f"{actual_score:.3f}", f"{m:.3f}", f"{s_dist:.3f}", f"{actual_score-s_dist:.2f}", f"{actual_score+s_dist:.2f}"]
    }))

# ----------------------------
# 3. COUNTY CONTEXT & RANKINGS
# ----------------------------
st.divider()
st.header("🌎 County-Wide Impact Ranking")

col_cdf, col_table = st.columns([2, 1])
with col_cdf:
    st.subheader("Cumulative Distribution (CDF)")
    st.write("The CDF curve ranks all 2,000+ tracts from least to most vulnerable. The yellow region indicates the Interquartile Range (Middle 50%).")
    all_tracts = df_comb[['s_e','s_i','s_h','s_s']].sum(axis=1).sort_values().values
    q1, q3 = np.percentile(all_tracts, 25), np.percentile(all_tracts, 75)
    fig_cdf, ax_cdf = plt.subplots(figsize=(10, 4.5))
    ax_cdf.plot(all_tracts, color='#2980b9', lw=3)
    idx_q1 = np.searchsorted(all_tracts, q1)
    idx_q3 = np.searchsorted(all_tracts, q3)
    ax_cdf.fill_between(range(idx_q1, idx_q3), all_tracts[idx_q1:idx_q3], color='#f1c40f', alpha=0.3, label='Interquartile Range')
    rank_idx = np.searchsorted(all_tracts, actual_score)
    ax_cdf.scatter(rank_idx, actual_score, color='red', s=120, zorder=5, label=f'ZIP {zip_in}')
    ax_cdf.legend()
    st.pyplot(fig_cdf)
    st.success(f"Ranking: ZIP {zip_in} has higher impact potential than **{ (all_tracts < actual_score).mean() * 100:.1f}%** of LA.")

with col_table:
    st.subheader("Mean Impact by ZIP")
    st.dataframe(zip_summary, height=450, hide_index=True)

# ----------------------------
# 4. PILLAR DEEP-DIVE (SORTED BY DRIVER)
# ----------------------------
st.divider()
st.header("🔍 Pillar Deep-Dive")
st.write("We have defined the **'Danger Zone'** for each individual pillar as any value exceeding **1 Standard Deviation** from the County Mean (Higher Need).")

def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True):
    sub = df[df['GEOID10'] == target_geoid]
    if sub.empty: return
    val = sub[col].values[0]
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(name)
        st.write(desc)
        st.metric(f"ZIP {zip_in}", f"{val:,.1f} {unit}")
        st.table(pd.DataFrame({
            "Metric": ["County Mean", "+1 SD (Danger)", "-1 SD"],
            "Value": [f"{mean_v:,.2f}", f"{mean_v + std_v:,.2f}", f"{mean_v - std_v:,.2f}"]
        }))
    with col2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        counts, edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        for i in range(len(patches)):
            mid = (edges[i] + edges[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        x = np.linspace(data.min(), data.max(), 500)
        ax.plot(x, norm.pdf(x, mean_v, std_v), color='black', lw=2)
        ax.axvline(val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(mean_v + std_v, color='red', ls=':', label='+1 SD')
        ax.axvline(mean_v - std_v, color='red', ls=':', label='-1 SD')
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)
    st.divider()

# Pillars ordered by their specific contribution to the local score
pillar_data = [
    (df_ejsm, 'CIscore', 'Environmental Justice', 'Points', "Composite of pollution exposure and biological vulnerability.", 's_e', 20, False, raw_scores['s_e']),
    (df_income, 'med_hh_income', 'Median HH Income', '$USD', "Inverted metric; lower income creates higher garden ROI.", 's_i', 250, False, raw_scores['s_i']),
    (df_heat, 'DegHourDay', 'Heat Burden', 'Days', "Degree-hour days above baseline; target for cooling.", 's_h', 150, True, raw_scores['s_h']),
    (df_snap, 'SNAP_pct', 'Food Access', '% Pop', "Households utilizing SNAP; proxy for food sovereignty.", 's_s', 150, True, raw_scores['s_s'])
]
for p in sorted(pillar_data, key=lambda x: x[8], reverse=True):
    plot_pillar(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
