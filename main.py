import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="PMR Sandbox (unofficial)", layout="wide")
st.title("PMR Sandbox (unofficial)")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df = load_data()

medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]

low_level_indicators = [col for col in df.columns if col not in ["Country", "OECD", "GDP_PCAP_2023", "PMR_2023"] + medium_level_indicators]

st.sidebar.header("Navigation Mode")
mode = st.sidebar.radio("Choose simulation mode:", ["Optimized", "Autonomous (hierarchical)"])

countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=0)

pmr_score = df[df["Country"] == selected_country]["PMR_2023"].values[0]
global_pct = (df["PMR_2023"] > pmr_score).mean() * 100

# Indicadores generales
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
with col2:
    st.metric(label="Global Percentile", value=f"{round(global_pct)}%", help="Relative to all countries in the dataset")

# Radar chart: país vs promedio OCDE
st.subheader("📊 PMR Profile: Country vs OECD Average (Medium-level indicators)")
row = df[df["Country"] == selected_country].iloc[0]
oecd_avg = df[df["OECD"] == 1][medium_level_indicators].mean()
country_vals = row[medium_level_indicators]

radar_fig = go.Figure()
radar_fig.add_trace(go.Scatterpolar(r=country_vals.values,
                                    theta=medium_level_indicators,
                                    fill='toself',
                                    name=selected_country,
                                    line=dict(color='blue')))
radar_fig.add_trace(go.Scatterpolar(r=oecd_avg.values,
                                    theta=medium_level_indicators,
                                    fill='toself',
                                    name='OECD Average',
                                    line=dict(color='gray')))
radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])),
                        showlegend=True)
st.plotly_chart(radar_fig, use_container_width=True)

# Modo optimizado
if mode == "Optimized":
    st.subheader("🔎 Regulatory Subcomponent Overview – Current Position by Percentile")
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        percentile = (df[ind] > score).mean() * 100
        if percentile > 80:
            level = "🔴 High"
        elif percentile > 60:
            level = "🟠 Medium"
        else:
            level = "🟢 Low"
        summary.append({"Indicator": ind, "Score": round(score, 2), "Percentile": round(percentile), "Level": level})

    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values("Percentile", ascending=False)
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.subheader("📌 Suggested Reform Priorities")
    impacts = []
    for ind in low_level_indicators:
        current = row[ind]
        percentile = (df[ind] > current).mean() * 100
        impacts.append({
            "indicator": ind,
            "score": current,
            "percentile": percentile
        })

    impacts_sorted = sorted(impacts, key=lambda x: x["percentile"], reverse=True)
    top3 = impacts_sorted[:3]

    sliders = {}
    for item in top3:
        st.markdown(f"**{item['indicator']}**\n\nCurrent score: {round(item['score'],2)} | Percentile: {round(item['percentile'])}%")
        sliders[item['indicator']] = st.slider(f"{item['indicator']}", 0.0, 6.0, float(item['score']), 0.1)

    simulated_row = row.copy()
    for ind, val in sliders.items():
        simulated_row[ind] = val

    # Recalcular PMR simulado para todos los países
    df["PMR_simulated"] = df[medium_level_indicators].mean(axis=1)
    
    # Ahora comparar el nuevo del país simulado contra esa distribución
    new_medium_avg = simulated_row[medium_level_indicators].mean()
    new_percentile = (df["PMR_simulated"] > new_medium_avg).mean() * 100

    original_medium = row[medium_level_indicators].mean()

    st.write("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Original PMR Estimate", round(original_medium, 3))
    with col5:
        st.metric("Simulated PMR Estimate", round(new_medium_avg, 3), delta=round(new_medium_avg - original_medium, 3))
    with col6:
        st.metric("Simulated Percentile", f"{round(new_percentile)}%")

else:
    st.info("Hierarchical simulation mode coming soon.")

