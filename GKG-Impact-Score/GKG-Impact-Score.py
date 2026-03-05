import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("🌿 Greenspace Impact Monte Carlo Dashboard")

st.markdown("""
This dashboard estimates the **impact score of a census tract**
based on environmental justice, income, SNAP usage, and heat burden.
A Monte Carlo simulation randomly weights these factors to account
for uncertainty in how they might be prioritized.
""")

# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    df_ejsm = pd.read_csv("GKG-Impact-Score/data/EJSM_DangerZone.csv")
    df_income = pd.read_csv("GKG-Impact-Score/data/Income_DangerZone.csv")
    df_snap = pd.read_csv("GKG-Impact-Score/data/TractSNAP_DangerZone.csv")
    df_heat = pd.read_csv("GKG-Impact-Score/data/DegHourDays_DangerZone.csv")
    df_ziptract = pd.read_excel("GKG-Impact-Score/data/ZIP_TRACT_122025.xlsx")
    return df_ejsm, df_income, df_snap, df_heat, df_ziptract

df_ejsm, df_income, df_snap, df_heat, df_ziptract = load_data()

# ----------------------------
# User Input
# ----------------------------

zip_used = st.text_input("Enter ZIP Code:", "90001")

# Clean ZIP-TRACT
df_ziptract['ZIP'] = df_ziptract['ZIP'].astype(str).str.zfill(5)
df_ziptract['GEOID10'] = df_ziptract['TRACT'].astype(str)

matching = df_ziptract[df_ziptract['ZIP'] == zip_used]

if matching.empty:
    st.error("ZIP code not found.")
    st.stop()

target_geoid = matching.iloc[0]['GEOID10']

# ----------------------------
# Standardize columns
# ----------------------------

def standardize(df, col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df_ejsm['CIscore_std'] = standardize(df_ejsm, 'CIscore')
df_income['med_hh_income_std'] = standardize(df_income, 'med_hh_income')
df_snap['TractSNAP_std'] = standardize(df_snap, 'TractSNAP')
df_heat['DegHourDay_std'] = standardize(df_heat, 'DegHourDay')

df_ejsm = df_ejsm.rename(columns={'Tract_1': 'GEOID10'})
df_income = df_income.rename(columns={'tract': 'GEOID10'})
df_snap = df_snap.rename(columns={'CT10': 'GEOID10'})
df_heat = df_heat.rename(columns={'FIPS': 'GEOID10'})

df_combined = df_ejsm[['GEOID10','CIscore_std']] \
    .merge(df_income[['GEOID10','med_hh_income_std']], on='GEOID10', how='outer') \
    .merge(df_snap[['GEOID10','TractSNAP_std']], on='GEOID10', how='outer') \
    .merge(df_heat[['GEOID10','DegHourDay_std']], on='GEOID10', how='outer') \
    .fillna(0)

x = df_combined[['CIscore_std','med_hh_income_std','TractSNAP_std','DegHourDay_std']].to_numpy()

# ----------------------------
# Monte Carlo
# ----------------------------

num_boot = 5000
coefficients = np.random.uniform(0,1,(num_boot,4))
coefficients /= coefficients.sum(axis=1, keepdims=True)

y = np.dot(coefficients, x.T)

ymed = np.median(y, axis=0)

# ----------------------------
# Plot 1
# ----------------------------

st.subheader("📊 Regional Impact Ranking")

st.markdown("""
This chart shows the **median weighted impact score**
for every census tract in the dataset.
The shaded region shows uncertainty from the Monte Carlo simulation.
""")

sorted_scores = np.sort(ymed)

fig1, ax1 = plt.subplots()
ax1.plot(sorted_scores)
ax1.set_ylabel("Impact Score")
ax1.set_xlabel("Census Tracts (Sorted)")
st.pyplot(fig1)

# ----------------------------
# Plot 2
# ----------------------------

st.subheader("🎯 ZIP Code Impact Distribution")

indices = np.where(df_combined['GEOID10'].to_numpy() == target_geoid)[0]

if len(indices) > 0:
    geo_distribution = y[:, indices[0]]
    mu = np.mean(geo_distribution)
    sigma = np.std(geo_distribution)

    st.markdown(f"""
    This histogram shows the uncertainty distribution
    for ZIP **{zip_used}**.

    Mean Impact Score: **{mu:.3f}**  
    Standard Deviation: **{sigma:.3f}**
    """)

    fig2, ax2 = plt.subplots()
    ax2.hist(geo_distribution, bins=40, density=True)
    ax2.axvline(mu)
    ax2.set_xlabel("Impact Score")
    st.pyplot(fig2)
else:
    st.warning("Selected ZIP not found in merged dataset.")





