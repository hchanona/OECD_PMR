import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PMR Sandbox (unofficial)", layout="wide")
st.title("PMR Sandbox (unofficial)")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

df = load_data()

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

indicators = df.columns[2:-1]  # exclude Country, OECD, GDP

def get_country_vector(df, country):
    row = df[df["Country"] == country]
    return row[indicators].T.rename(columns={row.index[0]: "Value"})

data_country = get_country_vector(df, selected_country)
data_benchmark = get_country_vector(df, benchmark_country)

pmr_score = df[df["Country"] == selected_country]["PMR_2023"].values[0]
gdp_score = df[df["Country"] == selected_country]["GDP_PCAP_2023"].values[0]
benchmark_score = df[df["Country"] == benchmark_country]["PMR_2023"].values[0]

# Percentil global
global_pct = (df["PMR_2023"] > pmr_score).mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
with col2:
    st.metric(label="GDP per capita (2023, PPP)", value=f"${round(gdp_score):,}")
with col3:
    st.metric(label="Global Percentile", value=f"{round(global_pct)}%", help="Relative to all countries in the dataset")

# Mostrar los tres rubros con peor desempeño
st.subheader("Top 3 Most Restrictive Subcomponents")
data_sorted = data_country.sort_values("Value", ascending=False).copy()
top_indicators = data_sorted.head(3).index.tolist()
cols = st.columns(3)
for i, ind in enumerate(top_indicators):
    val = data_sorted.loc[ind, "Value"]
    pct = (df[ind] > val).mean()*100
    cols[i].markdown(f"**{ind}**")
    cols[i].write(f"Score: {round(val,2)}")
    cols[i].write(f"Global percentile: {round(pct)}%")

# Visualizar subcomponentes
st.subheader("Subcomponent Scores")
data_plot = data_country.copy()
data_plot["Component"] = data_plot.index
fig = px.bar(data_plot.sort_values("Value", ascending=True), x="Value", y="Component", orientation="h",
             color="Value", color_continuous_scale="Reds",
             labels={"Value": "Score", "Component": "Indicator"})
st.plotly_chart(fig, use_container_width=True)

# Comparación radar con país de referencia
st.subheader("Regulatory Profile Comparison")
rdata = go.Figure()
rdata.add_trace(go.Scatterpolar(r=data_country.loc[indicators, "Value"],
                                theta=indicators,
                                fill='toself', name=selected_country))
rdata.add_trace(go.Scatterpolar(r=data_benchmark.loc[indicators, "Value"],
                                theta=indicators,
                                fill='toself', name=benchmark_country, opacity=0.6))
rdata.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])),
                    showlegend=True)
st.plotly_chart(rdata, use_container_width=True)

# Contexto comparativo con ingreso similar
st.subheader("Countries with Similar Income")
low_bound = gdp_score * 0.9
high_bound = gdp_score * 1.1
gdp_peers = df[(df["GDP_PCAP_2023"] >= low_bound) & (df["GDP_PCAP_2023"] <= high_bound)]
st.dataframe(gdp_peers[["Country", "GDP_PCAP_2023", "PMR_2023"]].sort_values("GDP_PCAP_2023"))

# Distribución PMR vs ingreso con línea de regresión
st.subheader("Distribution of PMR vs Income")
fig2 = px.scatter(df, x="GDP_PCAP_2023", y="PMR_2023", text="Country",
                  color=df["OECD"].map({1: "OECD", 0: "Non-OECD"}),
                  trendline="ols",
                  labels={"GDP_PCAP_2023": "GDP per capita (2023, PPP)", "PMR_2023": "PMR Score"})
fig2.update_traces(textposition='top center')
st.plotly_chart(fig2, use_container_width=True)

# Simulación de reforma
st.subheader("Simulate Reforms")
simulated = data_country.copy()
sliders = {}
for ind in top_indicators:
    val = data_country.loc[ind, "Value"]
    sliders[ind] = st.slider(f"{ind}", min_value=0.0, max_value=6.0, value=float(val), step=0.1)
    simulated.loc[ind, "Value"] = sliders[ind]

original_avg = data_country["Value"].mean()
simulated_avg = simulated["Value"].mean()
simulated_percentile = (df["PMR_2023"] > simulated_avg).mean()*100

st.write("---")
col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Original PMR Estimate", round(original_avg, 3))
with col5:
    st.metric("Simulated PMR Estimate", round(simulated_avg, 3), delta=round(simulated_avg-original_avg,3))
with col6:
    st.metric("Simulated Percentile", f"{round(simulated_percentile)}%")

st.success("This sandbox is an unofficial exploratory tool using publicly available PMR and World Bank data.")
