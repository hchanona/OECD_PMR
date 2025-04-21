import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configurar la app
st.set_page_config(page_title="PMR Reform Navigator", layout="wide")
st.title("PMR Reform Navigator")

# Cargar datos combinados con PIB per cápita
@st.cache_data
def load_data():
    df = pd.read_excel("data/PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df = load_data()

# Sidebar: selección de grupo de país y país
st.sidebar.header("Explore the PMR Indicators")
group_filter = st.sidebar.radio("Select country group", ["All", "OECD", "Non-OECD"])

if group_filter == "OECD":
    df_filtered = df[df["OECD"] == 1]
elif group_filter == "Non-OECD":
    df_filtered = df[df["OECD"] == 0]
else:
    df_filtered = df.copy()

countries = df_filtered["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Chile") if "Chile" in countries else 0)
benchmark_country = st.sidebar.selectbox("Compare to", df["Country"].tolist(), index=df["Country"].tolist().index("Denmark") if "Denmark" in df["Country"].tolist() else 1)

# Obtener datos por país
indicators = df.columns[2:-1]  # excluye Country, OECD y GDP

def get_country_vector(df, country):
    row = df[df["Country"] == country]
    return row[indicators].T.rename(columns={row.index[0]: "Value"})

data_country = get_country_vector(df, selected_country)
data_benchmark = get_country_vector(df, benchmark_country)

# Mostrar PMR general y PIB per cápita
pmr_score = df[df["Country"] == selected_country]["PMR_2023"].values[0]
gdp_score = df[df["Country"] == selected_country]["GDP_PCAP_2023"].values[0]
benchmark_score = df[df["Country"] == benchmark_country]["PMR_2023"].values[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
with col2:
    st.metric(label=f"GDP per capita (2023, PPP)", value=f"${round(gdp_score):,}")
with col3:
    st.metric(label=f"{benchmark_country} PMR Score", value=round(benchmark_score, 3))

# Visualizar subcomponentes
st.subheader("Subcomponent Scores")
data_plot = data_country.copy()
data_plot["Component"] = data_plot.index
fig = px.bar(data_plot.sort_values("Value", ascending=True), x="Value", y="Component", orientation="h",
             color="Value", color_continuous_scale="Reds",
             labels={"Value": "Score", "Component": "Indicator"})
st.plotly_chart(fig, use_container_width=True)

# Contexto comparativo con ingreso similar
st.subheader("Contextual Guidance by Income")

selected_gdp = gdp_score
low_bound = selected_gdp * 0.9
high_bound = selected_gdp * 1.1
gdp_peers = df[(df["GDP_PCAP_2023"] >= low_bound) & (df["GDP_PCAP_2023"] <= high_bound)]

st.write(f"Countries with similar GDP per capita (+/-10%) to {selected_country}:")
st.dataframe(gdp_peers[["Country", "GDP_PCAP_2023", "PMR_2023"]].sort_values("GDP_PCAP_2023"))

# Análisis de distribución del PMR por grupo de ingreso
st.subheader("Distribution of PMR vs Income")
fig2 = px.scatter(df, x="GDP_PCAP_2023", y="PMR_2023", text="Country",
                  color=df["OECD"].map({1: "OECD", 0: "Non-OECD"}),
                  labels={"GDP_PCAP_2023": "GDP per capita (2023, PPP)", "PMR_2023": "PMR Score"})
fig2.update_traces(textposition='top center')
st.plotly_chart(fig2, use_container_width=True)

# Sugerencia si el país está por encima del promedio del grupo
group_avg = df_filtered["PMR_2023"].mean()
if pmr_score > group_avg:
    st.warning(f"{selected_country}'s PMR score is above the average for {group_filter} countries ({round(group_avg,2)}). Consider reforming high-impact components below.")

# Simulación de reforma
st.subheader("Simulate Reforms")
top_indicators = data_plot.sort_values("Value", ascending=False).head(3).index.tolist()
simulated = data_country.copy()
sliders = {}
for ind in top_indicators:
    val = data_country.loc[ind, "Value"]
    sliders[ind] = st.slider(f"{ind}", min_value=0.0, max_value=6.0, value=float(val), step=0.1)
    simulated.loc[ind, "Value"] = sliders[ind]

# Resultado simulado
original_avg = data_country["Value"].mean()
simulated_avg = simulated["Value"].mean()
st.write("---")
col4, col5 = st.columns(2)
with col4:
    st.metric("Estimated Original Average", round(original_avg, 3))
with col5:
    st.metric("Estimated New Average", round(simulated_avg, 3), delta=round(simulated_avg - original_avg, 3))

st.success("App ready for policy exploration. Powered by PMR and World Bank data.")
