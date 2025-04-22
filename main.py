import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="PMR Sandbox (unofficial)", layout="wide")
st.title("PMR Sandbox (unofficial)")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df = load_data()

# Variables para agrupación de indicadores
medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]

low_level_indicators = [col for col in df.columns if col not in ["Country", "OECD", "GDP_PCAP_2023", "PMR_2023"] + medium_level_indicators]

# Interfaz
st.sidebar.header("Navigation Mode")
mode = st.sidebar.radio("Choose simulation mode:", ["Optimized", "Autonomous (hierarchical)"])

countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Chile") if "Chile" in countries else 0)

# Métricas básicas
pmr_score = df[df["Country"] == selected_country]["PMR_2023"].values[0]
gdp_score = df[df["Country"] == selected_country]["GDP_PCAP_2023"].values[0]
global_pct = (df["PMR_2023"] > pmr_score).mean() * 100

col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
with col2:
    oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
    non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
    st.metric(label='OECD Average PMR', value=round(oecd_avg, 3))
    st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3))

# Radar chart
st.subheader("📊 PMR Profile: Country vs OECD Average (Medium-level indicators)")
row = df[df["Country"] == selected_country].iloc[0]
oecd_avg_vals = df[df["OECD"] == 1][medium_level_indicators].mean()
country_vals = row[medium_level_indicators]

radar_fig = go.Figure()
radar_fig.add_trace(go.Scatterpolar(r=country_vals.values, theta=medium_level_indicators, fill='toself', name=selected_country, line=dict(color='blue')))
radar_fig.add_trace(go.Scatterpolar(r=oecd_avg_vals.values, theta=medium_level_indicators, fill='toself', name='OECD Average', line=dict(color='gray')))
radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=True)
st.plotly_chart(radar_fig, use_container_width=True)

# === Modo Optimizado ===
if mode == "Optimized":
    st.subheader("🔎 Regulatory Subcomponent Overview – Current Position by Percentile")
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        percentile = (df[ind] > score).mean() * 100
        if percentile > 90:
            level = "🔴 High"
        elif percentile > 50:
            level = "🟠 Medium"
        else:
            level = "🟢 Low"
        summary.append({"Indicator": ind, "Score": round(score, 2), "Percentile": round(percentile), "Level": level})

    df_summary = pd.DataFrame(summary).sort_values("Percentile", ascending=False)
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    # Reformas sugeridas
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

    # Calcular nuevo PMR (nivel medio) para el país simulado
    new_medium_avg = simulated_row[medium_level_indicators].mean()
    original_medium = row[medium_level_indicators].mean()

    # Insertar el valor simulado en una copia del df y comparar
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

else:
    st.info("Hierarchical simulation mode coming soon.")

# === Sidebar: Regresión ===
st.sidebar.markdown('### 📈 PMR Trends')
st.sidebar.subheader("🔎 PMR Score vs. GDP per capita & OECD Membership")
st.sidebar.write("""Este análisis estudia cómo el **ingreso per cápita** y la **pertenencia a la OCDE** afectan el **puntaje PMR** de un país.""")

# Preparar regresión cuadrática
X = df[["GDP_PCAP_2023", "OECD"]]
y = df["PMR_2023"]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = sm.OLS(y, X_poly).fit()

# Mostrar resultados
st.sidebar.write(model.summary())

# Gráfico con línea de regresión
st.sidebar.subheader("📊 Distribución de PMR vs Ingreso per cápita con Línea de Regresión")
fig = px.scatter(df, x="GDP_PCAP_2023", y="PMR_2023", text="Country", title="PMR vs Income per Capita", labels={"GDP_PCAP_2023": "Income per capita (PPP)", "PMR_2023": "PMR Score"})
fig.update_traces(textposition='top center')

x_vals = np.linspace(df["GDP_PCAP_2023"].min(), df["GDP_PCAP_2023"].max(), 100).reshape(-1, 1)
x_input = np.hstack([x_vals, np.full_like(x_vals, df["OECD"].mean())])  # Promedio OECD
x_poly_vals = poly.transform(x_input)
y_vals = model.predict(x_poly_vals)

fig.add_trace(go.Scatter(x=x_vals.flatten(), y=y_vals, mode='lines', name='Regresión cuadrática', line=dict(color='red', dash='dash')))
st.sidebar.plotly_chart(fig)

