import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# === VARIABLES ===
medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]
low_level_indicators = [col for col in df.columns if col not in ["Country", "OECD", "GDP_PCAP_2023", "PMR_2023"] + medium_level_indicators]

# === SIDEBAR ===
st.sidebar.header("Options")
mode = st.sidebar.radio("What do you want to do?", ["Guided simulation", "Autonomous simulation", "Stats"])

countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Chile") if "Chile" in countries else 0)

# === MODO: GUIDED SIMULATION ===
if mode == "Guided simulation":
    row = df[df["Country"] == selected_country].iloc[0]
    pmr_score = row["PMR_2023"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
    with col2:
        oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
        non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
        st.metric(label='OECD Average PMR', value=round(oecd_avg, 3))
        st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3))

    st.subheader("📊 PMR Profile: Country vs OECD Average (Medium-level indicators)")
    oecd_avg_vals = df[df["OECD"] == 1][medium_level_indicators].mean()
    country_vals = row[medium_level_indicators]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=country_vals.values, theta=medium_level_indicators, fill='toself', name=selected_country, line=dict(color='blue')))
    radar_fig.add_trace(go.Scatterpolar(r=oecd_avg_vals.values, theta=medium_level_indicators, fill='toself', name='OECD Average', line=dict(color='gray')))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("🔎 Regulatory Subcomponent Overview – Current Position by Percentile")
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        percentile = (df[ind] > score).mean() * 100
        level = "🔴 High" if percentile > 90 else "🟠 Medium" if percentile > 50 else "🟢 Low"
        summary.append({"Indicator": ind, "Score": round(score, 2), "Percentile": round(percentile), "Level": level})

    df_summary = pd.DataFrame(summary).sort_values("Percentile", ascending=False)
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.subheader("📌 Suggested Reform Priorities")
    top3 = df_summary.head(3)["Indicator"].tolist()

    sliders = {}
    for ind in top3:
        current = row[ind]
        percentile = (df[ind] > current).mean() * 100
        st.markdown(f"**{ind}**\n\nCurrent score: {round(current,2)} | Percentile: {round(percentile)}%")
        sliders[ind] = st.slider(ind, 0.0, 6.0, float(current), 0.1)

    simulated_row = row.copy()
    for ind, val in sliders.items():
        simulated_row[ind] = val

    new_medium_avg = simulated_row[medium_level_indicators].mean()
    original_medium = row[medium_level_indicators].mean()

    df_simulated = df.copy()
    df_simulated.loc[df_simulated["Country"] == selected_country, medium_level_indicators] = simulated_row[medium_level_indicators]
    df_simulated["PMR_simulated"] = df_simulated[medium_level_indicators].mean(axis=1)
    new_percentile = (df_simulated["PMR_simulated"] > new_medium_avg).mean() * 100

    st.write("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Original PMR Estimate", round(original_medium, 3))
    with col5:
        st.metric("Simulated PMR Estimate", round(new_medium_avg, 3), delta=round(new_medium_avg - original_medium, 3))
    with col6:
        st.metric("Simulated Percentile", f"{round(new_percentile)}%")

# === MODO: AUTONOMOUS SIMULATION ===
elif mode == "Autonomous simulation":
    st.info("🧭 Autonomous simulation mode coming soon. Stay tuned!")

# === MODO: STATS ===
elif mode == "Stats":
    st.header("📈 PMR Trends")

    st.subheader("🔎 PMR Score vs. GDP per capita (log-log) & OECD Membership")
    st.write("""
    Este análisis examina cómo el **ingreso per cápita (logarítmico)** y la **pertenencia a la OCDE** afectan el **logaritmo del PMR**. 
    Los coeficientes se interpretan como **elasticidades** o diferencias porcentuales aproximadas.
    """)

    # Preparar datos
    df_log = df[(df["PMR_2023"] > 0) & (df["GDP_PCAP_2023"] > 0)].copy()
    df_log["log_pmr"] = np.log(df_log["PMR_2023"])
    df_log["log_gdp"] = np.log(df_log["GDP_PCAP_2023"])

    # Entrenamiento del modelo
    X = sm.add_constant(df_log[["log_gdp", "OECD"]])
    y = df_log["log_pmr"]
    model = sm.OLS(y, X).fit()

    st.text("OLS Regression Results (log-log)")
    st.text(model.summary())

    # Predicción sobre datos nuevos
    st.subheader("📊 Distribución log(PMR) vs log(ingreso per cápita)")
    fig = px.scatter(df_log, x="log_gdp", y="log_pmr", text="Country",
                     labels={"log_gdp": "log(Income per capita)", "log_pmr": "log(PMR Score)"},
                     title="log(PMR) vs log(Income per capita)")
    x_vals = np.linspace(df_log["log_gdp"].min(), df_log["log_gdp"].max(), 100)

    X_pred = pd.DataFrame({
        "const": 1.0,
        "log_gdp": x_vals,
        "OECD": df_log["OECD"].mean()
    })

    y_vals = model.predict(X_pred)

    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name='Regresión lineal log-log',
            line=dict(color='red')
        )
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

