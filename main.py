import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="Product Market Regulator Sandbox", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("PMR_with_GDP.xlsx")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df["Country_clean"] = df["Country"].str.strip().str.lower()
    return df

df = load_data()

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

def compute_full_pmr(row, low_to_medium_map, medium_to_high_map):
    row = row.copy()

    for medium, lows in low_to_medium_map.items():
        values = [row[col] for col in lows if pd.notna(row[col])]
        row[medium] = np.mean(values) if values else np.nan

    for high, mediums in medium_to_high_map.items():
        values = [row[col] for col in mediums if pd.notna(row[col])]
        row[high] = np.mean(values) if values else np.nan

    high_values = [row[col] for col in medium_to_high_map.keys() if pd.notna(row[col])]
    row["PMR_simulated"] = np.mean(high_values) if high_values else np.nan

    return row

st.sidebar.markdown(
    "<h1 style='font-size: 28px; margin-bottom: 0;'>PMR Sandbox (Unofficial)</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("Options")
mode = st.sidebar.radio("What do you want to simulate?", ["Relative ranking", "Impact on the economy"])
countries = df["Country"].tolist()
selected_country = st.sidebar.selectbox("Select a country", countries, index=countries.index("Australia") if "Australia" in countries else 0)

if mode == "Relative ranking":
    st.markdown("""
    <h1 style='color:#1f77b4; font-weight: bold;'>PMR Sandbox – Relative Ranking</h1>
    """, unsafe_allow_html=True)

elif mode == "Impact on the economy":
    st.markdown("""
    <h1 style='color:#1f77b4; font-weight: bold;'>PMR Sandbox – Economic Impact</h1>
    """, unsafe_allow_html=True)


with st.expander("About this tool"):
    st.markdown("""
    This sandbox was designed as a **complementary interface** to the OECD’s official PMR Policy Simulator, which is available at: https://oecd-main.shinyapps.io/pmr-simulator/

    It aims to:
    - Show how PMR reform is economically relevant by illustrating its association with **GDP per capita** (via a log-log regression model).
    - Provide an intuitive first look into PMR structure using **medium- and low-level indicators**.

    For deeper exploration of the detailed components and underlying survey questions, the official OECD simulator remains the ideal next step.
    """)

st.sidebar.markdown("""---""")
st.sidebar.markdown("""
### What is the PMR Sandbox?

This tool allows you to explore and simulate the OECD’s Product Market Regulation (PMR) indicators. You can:
- Compare your country's regulatory profile with OECD and non-OECD averages,
- Simulate reforms to observe changes in your PMR ranking,
- Estimate the economic impact of better regulation on GDP per capita (PPP).

The PMR score ranges from 0 to 6. Lower scores are better — they indicate fewer regulatory barriers to competition and market access.

Data source: OECD PMR 2023–2024.
""")

if mode == "Relative ranking":
    selected_country_clean = selected_country.strip().lower()
    row = df[df["Country_clean"] == selected_country_clean].iloc[0]
    pmr_score = row["PMR_2023"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{selected_country} PMR Score", value=round(pmr_score, 3))
        global_rank_series = df["PMR_2023"].rank(method="min").astype(int)
        global_rank = int(global_rank_series.loc[df["Country_clean"] == selected_country_clean].values[0])
        st.metric(label="Rank among all countries", value=f"{global_rank} of {len(df)}")

    with col2:
        oecd_avg = df[df['OECD'] == 1]['PMR_2023'].mean()
        non_oecd_avg = df[df['OECD'] == 0]['PMR_2023'].mean()
        st.metric(label='OECD Average PMR', value=round(oecd_avg, 3))
        st.metric(label='Non-OECD Average PMR', value=round(non_oecd_avg, 3))

    st.subheader("PMR Profile: Country vs OECD Average (Medium-level indicators)")
    oecd_avg_vals = df[df["OECD"] == 1][medium_level_indicators].mean()
    country_vals = row[medium_level_indicators]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=country_vals.values, theta=medium_level_indicators, fill='toself', name=selected_country, line=dict(color='blue')))
    radar_fig.add_trace(go.Scatterpolar(r=oecd_avg_vals.values, theta=medium_level_indicators, fill='toself', name='OECD Average', line=dict(color='gray')))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,6])), showlegend=True)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("Regulatory Subcomponent Overview – Current Position by Rank")
    ranks = {ind: df[ind].rank(method="min").astype(int) for ind in low_level_indicators}
    rank_df = pd.DataFrame(ranks)
    summary = []
    for ind in low_level_indicators:
        score = row[ind]
        rank = int(rank_df[df["Country_clean"] == selected_country_clean][ind])
        summary.append({"Indicator": ind, "Score": round(score, 2) if pd.notna(score) else "N/A", "Rank": rank})

    df_summary = pd.DataFrame(summary).sort_values("Rank")
    st.dataframe(df_summary.reset_index(drop=True), use_container_width=True)

    st.subheader("Suggested Reform Priorities")
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

    if "PMR_simulated" not in df_simulated.columns:
        df_simulated["PMR_simulated"] = np.nan

    idx = df_simulated[df_simulated["Country_clean"] == selected_country_clean].index[0]
    for col in low_level_indicators + medium_level_indicators + high_level_indicators + ["PMR_simulated"]:
        df_simulated.at[idx, col] = simulated_row[col]

    df_simulated["PMR_simulated"] = df_simulated.apply(
        lambda row: compute_full_pmr(row, low_to_medium_map, medium_to_high_map)["PMR_simulated"], axis=1
    )

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
        
elif mode == "Impact on the economy":

    st.subheader("How is PMR associated with income per capita?")
    st.write("""This analysis estimates how a **reduction in PMR** is associated with a **percentage increase in GDP per capita (PPP)**.""")

    st.caption("""
    **Note:** GDP per capita (PPP) values correspond to the indicator *"GDP per capita, PPP (constant 2021 international $)"* from the **World Bank** (International Comparison Program,
    World Development Indicators database, Eurostat-OECD PPP Programme). Available at: https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD
    """)

    st.markdown("### Regression model")
    st.latex(r"\log(\text{GDP}_{\text{PCAP\_2023}}) = \beta_0 + \beta_1 \log(\text{PMR}_{2023}) + \beta_2 \cdot \text{OECD} + \varepsilon")

    df_log = df[(df["PMR_2023"] > 0) & (df["GDP_PCAP_2023"] > 0)].copy()
    df_log["log_pmr"] = np.log(df_log["PMR_2023"])
    df_log["log_gdp"] = np.log(df_log["GDP_PCAP_2023"])

    X = sm.add_constant(df_log[["log_pmr", "OECD"]])
    y = df_log["log_gdp"]
    model = sm.OLS(y, X).fit()

    st.subheader("Estimated elasticity")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Coef. log(PMR)", round(model.params["log_pmr"], 3))
    with col2:
        st.metric("p-value log(PMR)", f"{model.pvalues['log_pmr']:.3f}")
    with col3:
        st.metric("Adjusted R²", round(model.rsquared_adj, 3))

    st.markdown("---")
    st.subheader("Regression Plot: PMR vs GDP per capita (log-log)")

    scatter_fig = go.Figure()

    df_oecd = df_log[df_log["OECD"] == 1]
    scatter_fig.add_trace(go.Scatter(
    x=df_oecd["log_pmr"],
    y=df_oecd["log_gdp"],
    mode='markers',
    name="OECD Countries",
    marker=dict(color='blue', size=8, opacity=0.7),
    showlegend=True))

    df_non_oecd = df_log[df_log["OECD"] == 0]
    scatter_fig.add_trace(go.Scatter(
    x=df_non_oecd["log_pmr"],
    y=df_non_oecd["log_gdp"],
    mode='markers',
    name="Non-OECD Countries",
    marker=dict(color='red', size=8, opacity=0.7),
    showlegend=True))

    x_range = np.linspace(df_log["log_pmr"].min()-0.1, df_log["log_pmr"].max()+0.1, 100)
    beta0 = model.params["const"]
    beta1 = model.params["log_pmr"]
    beta2 = model.params["OECD"]

    y_oecd = beta0 + beta1 * x_range + beta2 * 1
    y_non_oecd = beta0 + beta1 * x_range + beta2 * 0

    scatter_fig.add_trace(go.Scatter(
    x=x_range,
    y=y_oecd,
    mode='lines',
    name=f"OECD Regression Line (slope={round(beta1,2)})",
    line=dict(color='blue', dash='dash')))

    scatter_fig.add_trace(go.Scatter(
    x=x_range,
    y=y_non_oecd,
    mode='lines',
    name=f"Non-OECD Regression Line (slope={round(beta1,2)})",
    line=dict(color='red', dash='dot')))

    scatter_fig.update_layout( xaxis_title="log(PMR)",yaxis_title="log(GDP per capita, PPP)", legend_title="Group", width=900, height=600,plot_bgcolor='white')

    st.plotly_chart(scatter_fig, use_container_width=True)


    st.markdown("---")
    st.subheader("Simulate the impact of PMR reform")
    selected_country_clean = selected_country.strip().lower()
    row = df[df["Country_clean"] == selected_country_clean].iloc[0]

    current_pmr = row["PMR_2023"]
    current_gdp = row["GDP_PCAP_2023"]
    is_oecd = row["OECD"]

    st.markdown(f"**{selected_country}** — Current PMR: **{round(current_pmr, 2)}**, Current GDP (PPP): **${round(current_gdp):,} USD**")

    pct_reduction = st.slider("% reduction in PMR", 0, 20, 10)
    new_pmr = current_pmr * (1 - pct_reduction / 100)

    log_pmr_now = np.log(current_pmr)
    log_pmr_new = np.log(new_pmr)
    delta_log_pmr = log_pmr_new - log_pmr_now

    coef = model.params["log_pmr"]
    delta_log_gdp = coef * delta_log_pmr
    pct_change_gdp = (np.exp(delta_log_gdp) - 1) * 100

    predicted_new_gdp = current_gdp * (1 + pct_change_gdp / 100)

    st.markdown(f"Reducing PMR from **{round(current_pmr, 2)}** to **{round(new_pmr, 2)}** is associated with an estimated **{round(pct_change_gdp, 2)}%** increase in GDP per capita.")
    st.metric("Projected GDP per capita (PPP)", f"${round(predicted_new_gdp):,} USD")

    pred_log_gdp_now = model.predict([[1.0, log_pmr_now, is_oecd]])[0]
    pred_gdp_now = np.exp(pred_log_gdp_now)

    if current_gdp > pred_gdp_now:
        st.markdown(f""" 
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 6px solid #ffeeba;">
        <b>Note:</b> {selected_country} currently has a GDP above the level predicted by the model for its PMR 
        (<b>${round(pred_gdp_now):,} USD</b> predicted vs <b>${round(current_gdp):,} USD</b> actual).
        <br><br>
        <em>Imagine what could happen if regulatory conditions improved further.</em>
        </div>
        """, unsafe_allow_html=True)

    st.caption("""
    *This simulator is based on an average elasticity estimated across all countries in the dataset. 
    The relationship shown is statistical, not causal, and may not apply directly to countries that already 
    perform significantly above or below model expectations.*
    """)

