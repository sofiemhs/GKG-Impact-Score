import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(layout="wide", page_title="GKG Impact Dashboard")

# --- SECTION 1: THE STORY & THE MATH ---
st.title("🌿 Good Karma Gardens: Impact Analysis Dashboard")
st.markdown("""
This dashboard identifies where lawn-to-garden conversions will have the highest community impact in Los Angeles County. 
We look at four "Pillars of Need" to calculate a single **Impact Score**.
""")

with st.expander("📖 How the Math Works (For Outsiders)"):
    st.write("### 1. Standardization")
    st.write("Because we can't add 'Dollars' to 'Heat Degrees,' we convert everything to a scale of **0.0 to 1.0**. A 1.0 means that tract is in the highest need category for that specific factor.")
    
    st.write("### 2. The Impact Equation")
    st.latex(r"Total Impact = EJSM_{std} + Income_{std} + Heat_{std} + SNAP_{std}")
    st.info("**Range:** 0.0 (Low Need) to 4.0 (Maximum Need).")
    
    st.write("### 3. Sensitivity (Monte Carlo)")
    st.write("We run 10,000 simulations where we randomly 'weight' different factors. This ensures our 'Street Status' is robust even if priorities change.")

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
# 2. LOCAL ANALYSIS (0.0 - 4.0 SCALE)
# ----------------------------

st.sidebar.header("📍 Set Location")
zip_in = st.sidebar.text_input("Enter LA County ZIP:", "91505")
match = df_ziptract[df_ziptract['ZIP'] == zip_in]

if match.empty:
    st.error(f"❌ ZIP {zip_in} is outside LA County or not in our dataset.")
    st.stop()

target_geoid = match.iloc[0]['GEOID10']
idx = np.where(df_comb['GEOID10'].values == target_geoid)[0]

if len(idx) > 0:
    # Scale Calculation: 0-4
    raw_pillar_scores = df_comb.iloc[idx[0]][['s_e','s_i','s_h','s_s']]
    actual_score = raw_pillar_scores.sum()
    
    # Monte Carlo Sensitivity (Scaled to 0-4)
    x_matrix = df_comb[['s_e','s_i','s_h','s_s']].to_numpy()
    weights = np.random.uniform(0, 1, (10000, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    sim_results = np.dot(weights, x_matrix.T) * 4 
    
    d = sim_results[:, idx[0]]
    m, s = norm.fit(d)
    
    # Identify Primary Driver (30% of Total Score)
    labels = {'s_e': 'EJSM', 's_i': 'Economic Need', 's_h': 'Heat', 's_s': 'Food Access'}
    drivers = [labels[k] for k, v in (raw_pillar_scores / actual_score).items() if v >= 0.30] if actual_score > 0 else []
    driver_text = f" & driven by **{', '.join(drivers)}**" if drivers else ""

    # Tiers for 0-4 Scale
    if actual_score < 0.8: tier, color, desc = "LOW IMPACT", "#2ecc71", "Healthy baseline metrics."
    elif 0.8 <= actual_score < 1.6: tier, color, desc = "MEDIUM IMPACT", "#f1c40f", "Emerging needs detected."
    elif 1.6 <= actual_score < 2.4: tier, color, desc = "HIGH IMPACT", "#e67e22", "Significant vulnerability."
    else: tier, color, desc = "VERY HIGH IMPACT", "#e74c3c", "DANGER ZONE: Critical priority."

    st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
        <p style="color:white; font-size:1.4rem; margin-top:5px; font-weight:bold;">{desc}{driver_text}</p></div>""", unsafe_allow_html=True)

    st.header(f"📊 Impact Sensitivity for {zip_in}")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(d, bins=30, color='#aed6f1', density=True, alpha=0.7, label='Simulation Distribution')
        x_range = np.linspace(min(d), max(d), 100)
        ax.plot(x_range, norm.pdf(x_range, m, s), color='#2e86c1', lw=3, label='Normal Curve Fit')
        ax.axvline(actual_score, color='#1b4f72', lw=3, label=f'Current ZIP: {actual_score:.2f}')
        ax.axvline(actual_score-s, color='#e74c3c', ls='--', label='-1 SD Bounds')
        ax.axvline(actual_score+s, color='#e74c3c', ls='--', label='+1 SD Bounds')
        ax.set_xlabel("Potential Total Impact Score (0.0 - 4.0)")
        ax.legend(fontsize='x-small')
        st.pyplot(fig)
    with col_r:
        st.markdown("### **The Result**")
        st.metric("Total Impact Score", f"{actual_score:.3f} / 4.0")
        st.table(pd.DataFrame({
            "Standard Deviation": [f"{s:.4f}"],
            "Confidence Range": [f"{actual_score-s:.2f} to {actual_score+s:.2f}"]
        }))
        st.caption("A small SD means the result is stable even if we prioritize different factors.")

# ----------------------------
# 3. COUNTY-WIDE RANKING (CDF)
# ----------------------------
st.divider()
st.header("🌎 Where does this ZIP rank in LA County?")
st.write("We ranked all 2,000+ census tracts in LA from lowest need to highest need.")

all_tract_sums = x_matrix.sum(axis=1)
sorted_sums = np.sort(all_tract_sums)
percentile = (all_tract_sums < actual_score).mean() * 100

fig_cdf, ax_cdf = plt.subplots(figsize=(12, 3))
ax_cdf.plot(sorted_sums, color='#2980b9', lw=3, label='LA County Need Curve')
ax_cdf.fill_between(range(len(sorted_sums)), sorted_sums, alpha=0.1, color='#3498db')
# Red dot placement
rank_idx = np.searchsorted(sorted_sums, actual_score)
ax_cdf.scatter(rank_idx, actual_score, color='red', s=100, zorder=5, label=f'ZIP {zip_in} Rank')
ax_cdf.set_title("Cumulative Distribution of Need (CDF)")
ax_cdf.set_ylabel("Impact Score (0-4)")
ax_cdf.set_xlabel("Tracts (Ranked Lowest to Highest)")
ax_cdf.legend()
st.pyplot(fig_cdf)
st.success(f"**Ranking Result:** ZIP {zip_in} has a higher need than **{percentile:.1f}%** of LA County.")



# ----------------------------
# 4. THE FOUR PILLARS (DEEP DIVE)
# ----------------------------
st.divider()
st.header("🔍 The Driving Factors")
st.write("Below is the 'Raw Data' for each pillar compared to the rest of the county.")

def plot_pillar(df, col, name, unit, desc, score_key, bins, is_high_danger=True):
    # SAFETY FIX: Check if ZIP exists in this specific pillar
    subset = df[df['GEOID10'] == target_geoid]
    if subset.empty:
        st.warning(f"⚠️ No local {name} data available for this specific tract.")
        return

    local_val = subset[col].values[0]
    data = df[col].dropna()
    mean_v, std_v = data.mean(), data.std()
    thresh = mean_v + std_v if is_high_danger else mean_v - std_v
    weight = df_comb.iloc[idx[0]][score_key]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(name)
        st.write(desc)
        st.metric(f"ZIP {zip_in} Value", f"{local_val:,.1f} {unit}")
        st.write(f"**Pillar Weight:** {weight:.3f} pts")
    with c2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        counts, bin_edges, patches = ax.hist(data, bins=bins, color='#bdc3c7', alpha=0.7, density=True)
        for i in range(len(patches)):
            mid = (bin_edges[i] + bin_edges[i+1]) / 2
            if (is_high_danger and mid > thresh) or (not is_high_danger and mid < thresh):
                patches[i].set_facecolor('#e74c3c')
        
        # Curve and Markers
        x = np.linspace(data.min(), data.max(), 500)
        ax.plot(x, norm.pdf(x, mean_v, std_v), color='black', lw=2, label='Normal Dist.')
        ax.axvline(local_val, color='blue', lw=3, label=f'ZIP {zip_in}')
        ax.axvline(mean_v, color='black', lw=1.5, label='County Mean')
        ax.axvline(mean_v - std_v, color='gray', ls=':', label='-1 SD')
        ax.axvline(mean_v + std_v, color='gray', ls=':', label='+1 SD')
        ax.set_title(f"{name} Distribution")
        ax.legend(fontsize='xx-small', ncol=2)
        st.pyplot(fig)



plot_pillar(df_ejsm, 'CIscore', 'Environmental Justice', 'Score', "Combines pollution and social vulnerability.", 's_e', 20, False)
st.divider()
plot_pillar(df_income, 'med_hh_income', 'Median Income', 'USD', "Financial health metric.", 's_i', 250, False)
st.divider()
plot_pillar(df_heat, 'DegHourDay', 'Heat Burden', 'Days', "Intensity of Urban Heat Island effect.", 's_h', 150, True)
st.divider()
plot_pillar(df_snap, 'SNAP_pct', 'Food Access', '% Pop', "Percentage of households needing food assistance.", 's_s', 150, True)
