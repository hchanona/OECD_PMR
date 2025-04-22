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

medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]

low_level_indicators = [col for col in df.columns if col not in ["Country", "OECD", "GDP_PCAP_2023", "PMR_2023"] + medium_level_indicators]

# Selección del modo de navegación
st.sidebar.header("Navigation Mode")
mode = st.sidebar.radio("Choose simulation mode:", ["Optimized", "Autonomous (hierarchical)"])

# Selección del país
countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Chile") if "Chile" in countries else 0)

# Obtener los puntajes de PMR y GDP para el país seleccionado
pmr_score = df[df["Country"] == selected_country]["PMR_2023"].values[0]
gdp_score = df[df["Country"] == selected_country]["GDP_PCAP_2023"].values[0]

# Cálculo del percentil global
global_pct = (df["PMR_2023"] > pmr_score).mean() * 100

# Mostrar las métricas principales
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
with col2:
    oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
    non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
    st.metric(label='OECD Average PMR', value=round(oecd_avg, 3), help='Average PMR for OECD countries')
    st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3), help='Average PMR for Non-OECD countries')

# Gráfico radar de la comparación entre el país seleccionado y el promedio de OCDE
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
        if percentile > 90:
            level = "🔴 High"
        elif percentile > 50:
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

# Apartado "PMR Trends" separado
st.markdown('### 📈 PMR Trends')

# Modo de regresión
st.subheader("🔎 PMR Score vs. GDP per capita & OECD Membership")
st.write("""
Este análisis estudia cómo el **ingreso per cápita** y la **pertenencia a la OCDE** afectan el **puntaje PMR** de un país.
""")

# Preparar las variables para la regresión
X = df[["GDP_PCAP_2023", "OECD"]]  # Variables independientes
X = sm.add_constant(X)  # Añadir constante (intercepto)
y = df["PMR_2023"]  # Variable dependiente

# Realizar la regresión
model = sm.OLS(y, X).fit()

# Mostrar resumen de los resultados de la regresión
st.write(model.summary())

# Sección de análisis gráfico
st.subheader("📊 Distribución de PMR vs Ingreso per cápita")
fig = px.scatter(df, x="GDP_PCAP_2023", y="PMR_2023", text="Country", title="PMR vs Income per Capita", labels={"GDP_PCAP_2023": "Income per capita (PPP)", "PMR_2023": "PMR Score"})
fig.update_traces(textposition='top center')
st.plotly_chart(fig)

