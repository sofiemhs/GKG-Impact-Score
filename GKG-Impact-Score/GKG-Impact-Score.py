# --- ADD THIS AFTER THE HERO STAT METRIC ---

# Define the Tier Logic
if m < 0.15:
    tier = "LOW IMPACT"
    color = "#2ecc71" # Green
    desc = "This area currently meets most environmental and social baseline metrics."
elif 0.15 <= m < 0.30:
    tier = "MEDIUM IMPACT"
    color = "#f1c40f" # Yellow
    desc = "This area shows emerging needs in heat or food access; a garden would be beneficial."
elif 0.30 <= m < 0.45:
    tier = "HIGH IMPACT"
    color = "#e67e22" # Orange
    desc = "This area has significant vulnerability. GKG builds here will have a major community effect."
else:
    tier = "VERY HIGH IMPACT"
    color = "#e74c3c" # Red
    desc = "DANGER ZONE: This tract is in the highest tier of need for green space conversion."

# Create a High-Visibility Status Bar
st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:0;">STREET STATUS: {tier}</h1>
        <p style="color:white; font-size:1.2rem; opacity:0.9;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# Add a visual Progress Bar to show where they sit on the scale
st.write(f"**Impact Scale Position:** {m:.3f} / 0.700")
st.progress(min(m / 0.7, 1.0))
