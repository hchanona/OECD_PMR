import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="Product Market Regulator Sandbox", layout="wide")
st.title("PMR Sandbox")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df["Country_clean"] = df["Country"].str.strip().str.lower()
    return df

df = load_data()
st.write(f"\U0001F30D This dataset includes **{df['Country'].nunique()} countries**.")

# === INDICADORES DEFINIDOS ===
high_level_indicators = [
    "Distortions Induced by State Involvement",
    "Barriers to Domestic and Foreign Entry"
]

medium_level_indicators = [
    "Distortions Induced by Public Ownership",
    "Involvement in Business Operations",
    "Regulations Impact Evaluation",
    "Administrative and Regulatory Burden",
    "Barriers in Service & Network sectors",
    "Barriers to Trade and Investment"
]

low_level_indicators = [
    col for col in df.columns
    if col not in ["Country", "Country_clean", "OECD", "GDP_PCAP_2023", "PMR_2023"]
    + medium_level_indicators
    + high_level_indicators
]

# === MAPEOS DE INDICADORES ===
low_to_medium_map = {
    "Distortions Induced by Public Ownership": [
        "Quality and Scope of Public Ownership", "Governance of SOEs"
    ],
    "Involvement in Business Operations": [
        "Retail Price Controls and Regulation",
        "Involvement in Business Operations in Network Sectors",
        "Involvement in Business Operations in Service Sectors",
        "Public Procurement"
    ],
    "Regulations Impact Evaluation": [
        "Assessment of Impact on Competition", "Interaction with Stakeholders"
    ],
    "Administrative and Regulatory Burden": [
        "Administrative Requirements for Limited Liability Companies and Personally-owned Enterprises",
        "Communication and Simplification of Administrative and Regulatory Burden"
    ],
    "Barriers in Service & Network sectors": [
        "Barriers to entry in Service Sectors", "Barriers to entry in Network Sectors"
    ],
    "Barriers to Trade and Investment": [
        "Barriers to FDI", "Barriers to Trade Facilitation", "Tariff Barriers"
    ]
}

medium_to_high_map = {
    "Distortions Induced by State Involvement": [
        "Distortions Induced by Public Ownership",
        "Involvement in Business Operations",
        "Regulations Impact Evaluation"
    ],
    "Barriers to Domestic and Foreign Entry": [
        "Administrative and Regulatory Burden",
        "Barriers in Service & Network sectors",
        "Barriers to Trade and Investment"
    ]
}

# === FUNCIÓN DE CÁLCULO COMPLETO DESDE NIVEL BAJO ===
def compute_full_pmr(row, low_to_medium_map, medium_to_high_map):
    row = row.copy()

    # Calcular medios desde bajos
    for medium, lows in low_to_medium_map.items():
        values = [row[col] for col in lows if pd.notna(row[col])]
        row[medium] = np.mean(values) if values else np.nan

    # Calcular altos desde medios
    for high, mediums in medium_to_high_map.items():
        values = [row[col] for col in mediums if pd.notna(row[col])]
        row[high] = np.mean(values) if values else np.nan

    # Calcular PMR desde altos (no se requiere mapeo porque son solo 2 componentes fijos)
    high_values = [row[col] for col in medium_to_high_map.keys() if pd.notna(row[col])]
    row["PMR_simulated"] = np.mean(high_values) if high_values else np.nan

    return row

# === SIDEBAR ===
st.sidebar.header("Options")
mode = st.sidebar.radio("What do you want to do?", ["Guided simulation", "Autonomous simulation", "Stats"])
countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Australia") if "Australia" in countries else 0)

# === MODO: SIMULACIÓN GUIADA ===
if mode == "Guided simulation":
    selected_country_clean = selected_country.strip().lower()
    row = df[df["Country_clean"] == selected_country_clean].iloc[0]
    pmr_score = row["PMR_2023"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
        # Rank global (entre todos los países)
        global_rank_series = df["PMR_2023"].rank(method="min").astype(int)
        global_rank = int(global_rank_series.loc[df["Country_clean"] == selected_country_clean].values[0])
        st.metric(label="Rank among all countries", value=f"{global_rank} of {len(df)}")

    with col2:
        oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
        non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
        st.metric(label='OECD Average PMR', value=round(oecd_avg, 3))
        st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3))

    st.subheader("\U0001F4CA PMR Profile: Country vs OECD Average (Medium-level indicators)")
    oecd_avg_vals = df[df["OECD"] == 1][medium_level_indicators].mean()
    country_vals = row[medium_level_indicators]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=country_vals.values, theta=medium_level_indicators, fill='toself', name=selected_country, line=dict(color='blue')))
    radar_fig.add_trace(go.Scatterpolar(r=oecd_avg_vals.values, theta=medium_level_indicators, fill='toself', name='OECD Average', line=dict(color='gray')))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("\U0001F50E Regulatory Subcomponent Overview – Current Position by Rank")
    ranks = {ind: df[ind].rank(method="min").astype(int) for ind in low_level_indicators}
    rank_df = pd.DataFrame(ranks)
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        rank = int(rank_df[df["Country_clean"] == selected_country_clean][ind])
        summary.append({"Indicator": ind, "Score": round(score, 2) if pd.notna(score) else "N/A", "Rank": rank})

    df_summary = pd.DataFrame(summary).sort_values("Rank")
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.subheader("\U0001F4CC Suggested Reform Priorities")
    top3 = df_summary.tail(3)["Indicator"].tolist()

    sliders = {}
    for ind in top3:
        current = row[ind]
        rank = int(rank_df[df["Country_clean"] == selected_country_clean][ind])
        st.markdown(f"**{ind}**\n\nCurrent score: {round(current,2)} | Rank: {rank}")
        sliders[ind] = st.slider(ind, 0.0, 6.0, float(current), 0.1)

    simulated_row = row.copy()
    for ind, val in sliders.items():
        simulated_row[ind] = val

    simulated_row = compute_full_pmr(simulated_row, low_to_medium_map, medium_to_high_map)

    new_medium_avg = simulated_row[medium_level_indicators].mean()
    original_medium = row[medium_level_indicators].mean()

    df_simulated = df.copy()

    # Asegura que la columna PMR_simulated exista
    if "PMR_simulated" not in df_simulated.columns:
        df_simulated["PMR_simulated"] = np.nan

    # Reemplazar los valores del país simulado
    idx = df_simulated[df_simulated["Country_clean"] == selected_country_clean].index[0]
    for col in low_level_indicators + medium_level_indicators + high_level_indicators + ["PMR_simulated"]:
        df_simulated.at[idx, col] = simulated_row[col]

    # Recalcular PMR_simulated para todos los países
    df_simulated["PMR_simulated"] = df_simulated.apply(
        lambda row: compute_full_pmr(row, low_to_medium_map, medium_to_high_map)["PMR_simulated"], axis=1
    )

    # Calcular ranking
    valid_simulated = df_simulated[df_simulated["PMR_simulated"].notna()].copy()
    valid_simulated["rank_simulated"] = valid_simulated["PMR_simulated"].rank(method="min")

    new_rank = int(valid_simulated.loc[valid_simulated["Country_clean"] == selected_country_clean, "rank_simulated"].values[0])

    st.write("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Original PMR Estimate", round(original_medium, 3))
    with col5:
        st.metric("Simulated PMR Estimate", round(new_medium_avg, 3), delta=round(new_medium_avg - original_medium, 3))
    with col6:
        st.metric("Simulated Rank", f"{new_rank}" if new_rank is not None else "N/A")

elif mode == "Autonomous simulation":
    st.subheader("🧭 Autonomous Simulation – Choose Reform Areas Hierarchically")

    selected_country_clean = selected_country.strip().lower()
    row = df[df["Country_clean"] == selected_country_clean].iloc[0]
    pmr_score = row["PMR_2023"]

    st.markdown(f"**{selected_country}** – Current PMR Score: **{round(pmr_score, 3)}**")

    # Paso 1: Selección de rubros
    selected_rubros = st.multiselect(
        "Selecciona hasta 3 rubros de nivel medio para simular reformas:",
        options=medium_level_indicators,
        max_selections=3,
        help="Puedes elegir hasta tres rubros para ajustar sus subcomponentes"
    )

    if selected_rubros:
        simulated_row = row.copy()
        st.write("### 🎯 Ajusta los subcomponentes de cada rubro seleccionado:")

        for rubro in selected_rubros:
            st.markdown(f"**{rubro}**")
            subcomponents = low_to_medium_map.get(rubro, [])
            if not subcomponents:
                st.warning("No se encontraron subcomponentes definidos para este rubro.")
                continue

            for sub in subcomponents:
                if sub in simulated_row:
                    current_val = simulated_row[sub]
                    new_val = st.slider(f"{sub}", 0.0, 6.0, float(current_val), 0.1)
                    simulated_row[sub] = new_val

        # Recalcular todo el PMR jerárquicamente
        simulated_row = compute_full_pmr(simulated_row, low_to_medium_map, medium_to_high_map)
        new_pmr = simulated_row["PMR_simulated"]

        # Clonar df y recalcular PMR_simulated para todos
        df_simulated = df.copy()
        if "PMR_simulated" not in df_simulated.columns:
            df_simulated["PMR_simulated"] = np.nan

        idx = df_simulated[df_simulated["Country_clean"] == selected_country_clean].index[0]
        for col in low_level_indicators + medium_level_indicators + high_level_indicators + ["PMR_simulated"]:
            df_simulated.at[idx, col] = simulated_row[col]

        df_simulated["PMR_simulated"] = df_simulated.apply(
            lambda row: compute_full_pmr(row, low_to_medium_map, medium_to_high_map)["PMR_simulated"], axis=1
        )

        # Ranking absoluto
        valid_simulated = df_simulated[df_simulated["PMR_simulated"].notna()].copy()
        valid_simulated["rank_simulated"] = valid_simulated["PMR_simulated"].rank(method="min")

        original_rank = int(df["PMR_2023"].rank(method="min").loc[df["Country_clean"] == selected_country_clean].values[0])
        new_rank = int(valid_simulated.loc[valid_simulated["Country_clean"] == selected_country_clean, "rank_simulated"].values[0])

        st.write("---")
        col7, col8, col9 = st.columns(3)
        with col7:
            st.metric("Original PMR", round(pmr_score, 3))
        with col8:
            st.metric("Simulated PMR", round(new_pmr, 3), delta=round(new_pmr - pmr_score, 3))
        with col9:
            st.metric("Simulated Rank", f"{new_rank} (original: {original_rank})")
    else:
        st.info("Selecciona al menos un rubro de nivel medio para comenzar.")

        
elif mode == "Stats":
    st.header("📈 PMR Impact Simulator")

    st.subheader("🔎 ¿Qué tan asociado está el PMR con el ingreso per cápita?")
    st.write("""
    Este análisis estima cómo una **reducción en el PMR** se asocia con un **aumento porcentual en el ingreso per cápita ajustado por paridad de compra** (PIB PPC).

    Se basa en una regresión lineal del logaritmo del PIB per cápita sobre el logaritmo del PMR y la membresía OCDE:

    `log(GDP_PCAP_2023) ~ log(PMR_2023) + OECD`
    """)

    # Preparar datos válidos
    df_log = df[(df["PMR_2023"] > 0) & (df["GDP_PCAP_2023"] > 0)].copy()
    df_log["log_pmr"] = np.log(df_log["PMR_2023"])
    df_log["log_gdp"] = np.log(df_log["GDP_PCAP_2023"])

    # Ajustar modelo
    X = sm.add_constant(df_log[["log_pmr", "OECD"]])
    y = df_log["log_gdp"]
    model = sm.OLS(y, X).fit()

    st.subheader("📉 Elasticidad estimada")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coef. log(PMR)", round(model.params["log_pmr"], 3))
    with col2:
        st.metric("Coef. OCDE", round(model.params["OECD"], 3))
    with col3:
        st.metric("R² ajustado", round(model.rsquared_adj, 3))

    st.markdown("---")
    st.subheader("🧮 Simula el impacto de mejorar el PMR")
    selected_country_clean = selected_country.strip().lower()
    row = df[df["Country_clean"] == selected_country_clean].iloc[0]

    current_pmr = row["PMR_2023"]
    current_gdp = row["GDP_PCAP_2023"]
    is_oecd = row["OECD"]

    st.markdown(f"**{selected_country}** — PMR actual: **{round(current_pmr, 2)}**, PIB PPC actual: **${round(current_gdp):,} USD**")

    # Slider: mejora en PMR (% reducción)
    pct_reduction = st.slider("% de reducción en el PMR", 0, 50, 10)
    new_pmr = current_pmr * (1 - pct_reduction / 100)

    # Calcular log-pmr antes y después
    log_pmr_now = np.log(current_pmr)
    log_pmr_new = np.log(new_pmr)
    delta_log_pmr = log_pmr_new - log_pmr_now

    # Usar coeficiente para estimar delta log(GDP)
    coef = model.params["log_pmr"]
    delta_log_gdp = coef * delta_log_pmr
    pct_change_gdp = (np.exp(delta_log_gdp) - 1) * 100

    predicted_new_gdp = current_gdp * (1 + pct_change_gdp / 100)

    st.markdown(f"Reducir el PMR de **{round(current_pmr, 2)}** a **{round(new_pmr, 2)}** está asociado a un incremento estimado del **{round(pct_change_gdp, 2)}%** en el PIB per cápita.")
    st.metric("PIB PPC proyectado", f"${round(predicted_new_gdp):,} USD")

    # Mostrar diferencia entre actual y predicho por el modelo
    pred_log_gdp_now = model.predict([[1.0, log_pmr_now, is_oecd]])[0]
    pred_gdp_now = np.exp(pred_log_gdp_now)

    if current_gdp > pred_gdp_now:
        st.warning(f"{selected_country} ya tiene un ingreso por encima de lo predicho por el modelo para su PMR actual (**${round(pred_gdp_now):,} USD** predicho vs **${round(current_gdp):,} USD** observado).")

    st.caption("""
    📌 *Este simulador se basa en una elasticidad promedio estimada para todos los países. La relación mostrada es estadística, no causal, y puede no aplicarse directamente a países que ya están significativamente por encima o por debajo del promedio.*
    """)
