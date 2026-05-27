import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import requests
import os
import numpy as np
import json
from urllib.request import urlopen
import scipy.stats as stats

dash.register_page(__name__, path='/march-2026-heat', title='March 2026 Heat Wave | Becky Bolinger')

current_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DATA LOADING (STATIONS & COUNTIES)
# ==========================================

# --- Station Data ---
try:
    df_stations = pd.read_csv(os.path.join(current_dir, '..', 'data', 'march_2026_heat_records.csv'))
    available_states = sorted(df_stations['state'].unique().tolist())
except FileNotFoundError:
    df_stations = pd.DataFrame(columns=["latitude", "longitude", "name", "state", "elevation", "GHCN_ID", "record_temp", "years_on_record"])
    available_states = []

# --- County Data ---
try:
    df_county = pd.read_csv(os.path.join(current_dir, '..', 'data', 'county_march_history.csv'))
    df_county['FIPS'] = df_county['FIPS'].astype(str).str.zfill(5)
    
    idx_max = df_county.groupby('FIPS')['Temp'].idxmax()
    df_max = df_county.loc[idx_max]
    record_fips = df_max[df_max['Year'] == 2026]['FIPS'].unique()
    
    df_map_county = df_max[df_max['FIPS'].isin(record_fips)].copy()
    
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties_geojson = json.load(response)
except FileNotFoundError:
    df_county = pd.DataFrame()
    df_map_county = pd.DataFrame()
    counties_geojson = {}

# --- Helper: ACIS Fetch for Stations ---
def get_heat_data(station_id):
    payload = {
        "sid": station_id, "sdate": "2026-03-01", "edate": "2026-03-31",
        "elems": [
            {"name": "maxt"},                            # 0: Obs Max
            {"name": "maxt", "normal": "1"},             # 1: Normal Max
            {"name": "maxt", "extreme": "max"},          # 2: Record High Max
            {"name": "maxt", "extreme": "min"},          # 3: Record Low Max
            {"name": "mint"},                            # 4: Obs Min
            {"name": "mint", "normal": "1"},             # 5: Normal Min
            {"name": "mint", "extreme": "max"},          # 6: Record High Min
            {"name": "mint", "extreme": "min"}           # 7: Record Low Min
        ]
    }
    try:
        r = requests.post("http://data.rcc-acis.org/StnData", json=payload, timeout=10)
        data = r.json()
        if "data" not in data: return None
        
        dates = []
        obs_max, norm_max, rec_high_max, rec_low_max = [], [], [], []
        obs_min, norm_min, rec_high_min, rec_low_min = [], [], [], []
        
        for item in data['data']:
            dates.append(item[0])
            
            def parse_val(v):
                try: return float(v)
                except: return np.nan
                
            obs_max.append(parse_val(item[1]))
            norm_max.append(parse_val(item[2]))
            rec_high_max.append(parse_val(item[3]))
            rec_low_max.append(parse_val(item[4]))
            obs_min.append(parse_val(item[5]))
            norm_min.append(parse_val(item[6]))
            rec_high_min.append(parse_val(item[7]))
            rec_low_min.append(parse_val(item[8]))
            
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "obs_max": obs_max, "norm_max": norm_max, "rec_high_max": rec_high_max, "rec_low_max": rec_low_max,
            "obs_min": obs_min, "norm_min": norm_min, "rec_high_min": rec_high_min, "rec_low_min": rec_low_min
        })
        return df
    except Exception as e:
        print(f"ACIS Error: {e}")
        return None

# ==========================================
# 2. LAYOUT
# ==========================================
layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("The March 2026 Extreme Heat Event", className="display-5 fw-bold text-danger mb-3"),
                    html.P("An interactive retrospective of the unprecedented early-season heat wave.", className="lead text-muted"),
                    html.Hr(className="my-4")
                ]
            )
        ),

        # --- SECTION 1: STATION LEVEL ---
        dbc.Row(dbc.Col(html.H3("1. Local Station Records", className="fw-bold mb-3 text-primary"), width=12)),
        dbc.Row(
            dbc.Col(
                [
                    html.Label("Filter Stations by State:", className="fw-bold"),
                    dcc.Dropdown(
                        id="heat-state-filter",
                        options=[{'label': s, 'value': s} for s in available_states],
                        placeholder="Select a State...",
                        clearable=True
                    )
                ],
                width=12, md=4, className="mb-3"
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="heat-map", style={"height": "550px"}, config={"scrollZoom": True}),
                    width=12, lg=5, className="mb-4"
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Daily Observed vs. Record Envelopes (March 2026)", className="fw-bold bg-light"),
                            dbc.CardBody(
                                dcc.Loading(
                                    dcc.Graph(id="heat-station-graph", style={"height": "500px"}),
                                    type="circle", color="#dc3545"
                                )
                            )
                        ],
                        className="shadow-sm border-0 h-100"
                    ),
                    width=12, lg=7, className="mb-4"
                )
            ]
        ),

        html.Hr(className="my-5 border-2"),

        # --- SECTION 2: COUNTY LEVEL ---
        dbc.Row(dbc.Col(html.H3("2. Regional County-Level Extremes", className="fw-bold mb-4 text-primary"), width=12)),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Counties with Record Warmest March (2026)", className="fw-bold bg-light"),
                                dbc.CardBody(dcc.Graph(id="county-map", style={"height": "400px"}, config={"scrollZoom": True}))
                            ],
                            className="shadow-sm border-0 mb-4"
                        ),
                        dbc.Alert(
                            id="return-period-text",
                            children="Click a county on the map to view its historical distribution and return period analysis.",
                            color="danger",
                            className="shadow-sm"
                        )
                    ],
                    width=12, lg=5
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="time-series-chart", style={"height": "250px"}, className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="dist-chart", style={"height": "250px"}), width=6),
                                dbc.Col(dcc.Graph(id="joint-chart", style={"height": "250px"}), width=6),
                            ]
                        )
                    ],
                    width=12, lg=7
                )
            ],
            className="mb-5"
        )
    ],
    fluid=True,
    className="py-4"
)

# ==========================================
# 3. CALLBACKS (STATION TOOL)
# ==========================================
@callback(
    Output("heat-map", "figure"),
    Input("heat-state-filter", "value")
)
def update_station_map(selected_state):
    dff = df_stations.copy()
    zoom, center = 3, {"lat": 39.5, "lon": -98.35}
    if selected_state:
        dff = dff[dff['state'] == selected_state]
        if not dff.empty:
            center = {"lat": dff['latitude'].mean(), "lon": dff['longitude'].mean()}
            zoom = 5.5

    fig = px.scatter_mapbox(
        dff, lat="latitude", lon="longitude", hover_name="name",
        custom_data=["GHCN_ID", "state", "record_temp", "years_on_record"], 
        color="record_temp", color_continuous_scale=px.colors.sequential.YlOrRd,
        size_max=15, zoom=zoom, center=center, mapbox_style="carto-positron",
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, uirevision=selected_state)
    return fig

@callback(
    Output("heat-station-graph", "figure"),
    Input("heat-map", "clickData")
)
def update_station_graph(clickData):
    if not clickData:
        return go.Figure().update_layout(xaxis={"visible":False}, yaxis={"visible":False}, annotations=[{"text": "Click a station on the map above...", "showarrow":False}])

    point = clickData['points'][0]
    station_id, station_name = point['customdata'][0], point['hovertext']
    df = get_heat_data(station_id)
    
    if df is None or df.empty: return go.Figure().update_layout(title="Data unavailable.")

    # Create a 2-Row Subplot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # --- ROW 1: MAXIMUM TEMPERATURES ---
    # Shaded Range: Record Low Max to Record High Max
    fig.add_trace(go.Scatter(
        x=df['date'].tolist() + df['date'].tolist()[::-1],
        y=df['rec_high_max'].tolist() + df['rec_low_max'].tolist()[::-1],
        fill='toself', fillcolor='rgba(220, 53, 69, 0.1)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="Record Max Envelope", legendgroup="max"
    ), row=1, col=1)
    
    # Normal Max Line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['norm_max'], mode='lines', 
        name='Normal Max', line=dict(color='gray', width=1.5, dash='dash'), legendgroup="max"
    ), row=1, col=1)
    
    # Observed Max Line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['obs_max'], mode='lines+markers', 
        name='Observed Max', line=dict(color='#dc3545', width=2.5), marker=dict(size=5), legendgroup="max"
    ), row=1, col=1)


    # --- ROW 2: MINIMUM TEMPERATURES ---
    # Shaded Range: Record Low Min to Record High Min
    fig.add_trace(go.Scatter(
        x=df['date'].tolist() + df['date'].tolist()[::-1],
        y=df['rec_high_min'].tolist() + df['rec_low_min'].tolist()[::-1],
        fill='toself', fillcolor='rgba(13, 110, 253, 0.1)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name="Record Min Envelope", legendgroup="min"
    ), row=2, col=1)
    
    # Normal Min Line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['norm_min'], mode='lines', 
        name='Normal Min', line=dict(color='gray', width=1.5, dash='dash'), legendgroup="min"
    ), row=2, col=1)
    
    # Observed Min Line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['obs_min'], mode='lines+markers', 
        name='Observed Min', line=dict(color='#0d6efd', width=2.5), marker=dict(size=5), legendgroup="min"
    ), row=2, col=1)

    # Header calculations (still using the Peak/Anom metrics)
    peak_temp = df['obs_max'].max()
    max_anomaly = (df['obs_max'] - df['norm_max']).max()
    
    title_text = f"<b>{station_name}</b><br><span style='font-size:13px;color:#555;'>Peak Temp: {peak_temp}°F | Max Anomaly: +{max_anomaly:.1f}°F</span>"

    fig.update_layout(
        title=title_text,
        template="simple_white", hovermode="x unified",
        margin=dict(t=60, b=30),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
    )
    
    fig.update_yaxes(title_text="Max Temp (°F)", row=1, col=1)
    fig.update_yaxes(title_text="Min Temp (°F)", row=2, col=1)
    fig.update_xaxes(title_text="March 2026", row=2, col=1)

    return fig

# ==========================================
# 4. CALLBACKS (COUNTY TOOL)
# ==========================================
@callback(
    Output("county-map", "figure"),
    Input("county-map", "id")
)
def draw_county_map(_):
    if df_map_county.empty: return go.Figure()
    fig = px.choropleth_mapbox(
        df_map_county, geojson=counties_geojson, locations='FIPS',
        color='Temp_Anomaly', color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron", zoom=3.5, center={"lat": 39.0, "lon": -96.0},
        opacity=0.8, hover_data={"FIPS": False, "Temp": True, "Temp_Anomaly": True}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@callback(
    Output("time-series-chart", "figure"),
    Output("dist-chart", "figure"),
    Output("joint-chart", "figure"),
    Output("return-period-text", "children"),
    Input("county-map", "clickData")
)
def update_county_charts(clickData):
    if not clickData or df_county.empty:
        empty = go.Figure().update_layout(xaxis={"visible":False}, yaxis={"visible":False}, template="simple_white")
        return empty, empty, empty, "Select a county on the map to begin analysis."

    fips = clickData['points'][0]['location']
    df_c = df_county[df_county['FIPS'] == fips].copy().sort_values('Year')
    if df_c.empty: return go.Figure(), go.Figure(), go.Figure(), "Data unavailable."

    row_2026 = df_c[df_c['Year'] == 2026]
    val_2026_temp = row_2026['Temp'].values[0] if not row_2026.empty else np.nan
    val_2026_pcp = row_2026['Precip'].values[0] if not row_2026.empty else np.nan

    # 1. Time Series
    fig_ts = go.Figure()
    colors = ['#dc3545' if val > 0 else '#0d6efd' for val in df_c['Temp_Anomaly']]
    fig_ts.add_trace(go.Bar(x=df_c['Year'], y=df_c['Temp_Anomaly'], marker_color=colors, name="Anomaly"))
    if not np.isnan(val_2026_temp):
        fig_ts.add_trace(go.Scatter(x=[2026], y=[row_2026['Temp_Anomaly'].values[0]], mode='markers', marker=dict(color='gold', size=10, line=dict(color='black', width=1))))
    fig_ts.update_layout(title="March Temp Anomaly (1895-2026)", margin=dict(t=30, b=10, l=10, r=10), template="simple_white", showlegend=False)

    # 2. Temp Distribution
    fig_dist = px.histogram(df_c, x="Temp", nbins=20, opacity=0.7, color_discrete_sequence=['gray'])
    if not np.isnan(val_2026_temp):
        fig_dist.add_vline(x=val_2026_temp, line_width=3, line_dash="dash", line_color="#dc3545")
        fig_dist.add_annotation(x=val_2026_temp, y=0.95, yref="paper", text="2026", showarrow=False, font=dict(color="#dc3545", size=12), xanchor="left")
    fig_dist.update_layout(title="Temp Distribution", xaxis_title="Temp (°F)", yaxis_title="Count", margin=dict(t=30, b=10, l=10, r=10), template="simple_white")

    # 3. Joint Probability
    fig_joint = go.Figure()
    fig_joint.add_trace(go.Scatter(x=df_c['Temp'], y=df_c['Precip'], mode='markers', marker=dict(color='gray', opacity=0.5)))
    if not np.isnan(val_2026_temp) and not np.isnan(val_2026_pcp):
        fig_joint.add_trace(go.Scatter(x=[val_2026_temp], y=[val_2026_pcp], mode='markers', marker=dict(color='#dc3545', size=10, line=dict(color='black', width=1))))
    fig_joint.update_layout(title="Temp vs. Precip", xaxis_title="Temp (°F)", yaxis_title="Precip (in)", margin=dict(t=30, b=10, l=10, r=10), template="simple_white", showlegend=False)

    # 4. Return Period
    mu, std = stats.norm.fit(df_c['Temp'])
    if not np.isnan(val_2026_temp):
        z_score = (val_2026_temp - mu) / std
        p_exceed = stats.norm.sf(z_score) 
        return_period_str = "> 10,000 years" if p_exceed < 0.0001 else f"{int(round(1 / p_exceed)):,} years"
        text_out = html.Span([
            html.B("Statistical Rarity: "),
            f"Fitted to a normal distribution, the March 2026 temperature of {val_2026_temp:.1f}°F has a Z-score of +{z_score:.2f}. This represents an estimated return frequency of a ",
            html.B(f"1-in-{return_period_str} event"), "."
        ])
    else: text_out = "Data for 2026 is missing."

    return fig_ts, fig_dist, fig_joint, text_out