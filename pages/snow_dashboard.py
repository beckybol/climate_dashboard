import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import requests
import datetime
import os
import numpy as np

dash.register_page(__name__, title='Snowfall Dashboard', path='/snow_dashboard')

# --- 1. SETUP & DATA LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

# Load Map Data
map_csv = os.path.join(data_dir, 'snow_stations_map_data.csv')
try:
    df_map = pd.read_csv(map_csv)
    available_states = sorted(df_map['state'].unique().tolist())
except FileNotFoundError:
    df_map = pd.DataFrame(columns=["latitude", "longitude", "annual_snow", "name", "state", "elevation", "GHCN_ID"])
    available_states = []

# Load Normals Data
normals_csv = os.path.join(data_dir, 'mly-snow-normal-custom-30yr.csv')
try:
    df_normals = pd.read_csv(normals_csv)
except FileNotFoundError:
    df_normals = pd.DataFrame()


# --- 2. HELPER: FETCH DAILY ACIS DATA ---
def get_daily_acis_data(station_id):
    """
    Fetches DAILY snowfall from July 1st to Present.
    Returns a processed DataFrame with daily values and flags.
    """
    # 1. Determine "Season Start" (July 1st)
    today = datetime.date.today()
    if today.month >= 7:
        start_year = today.year
    else:
        start_year = today.year - 1
        
    sdate = f"{start_year}-07-01"
    edate = today.strftime("%Y-%m-%d")

    # 2. API Call (Daily Data)
    payload = {
        "sid": station_id,
        "sdate": sdate,
        "edate": edate,
        "elems": [{"name": "snow"}] # Default is daily
    }

    try:
        r = requests.post("http://data.rcc-acis.org/StnData", json=payload, timeout=10)
        data = r.json()
        
        if "data" not in data:
            return None

        # 3. Process to DataFrame
        # ACIS returns: [["2025-07-01", "0.0"], ["2025-07-02", "T"], ...]
        dates = []
        vals = []
        flags = []
        
        for item in data['data']:
            d_str = item[0]
            v_str = item[1]
            
            # Parse Value
            if v_str == "M": 
                v = np.nan
                f = "Missing"
            elif v_str == "T": 
                v = 0.0001 # Trace
                f = "Trace"
            elif v_str == "S": # Subsequent (accumulated in next day)
                v = np.nan 
                f = "Subsequent"
            else:
                try:
                    v = float(v_str)
                    f = "Good"
                except:
                    v = np.nan
                    f = "Error"
            
            dates.append(d_str)
            vals.append(v)
            flags.append(f)

        df = pd.DataFrame({"date": dates, "snow": vals, "flag": flags})
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        
        return df
        
    except Exception as e:
        print(f"ACIS Error: {e}")
        return None


# --- 3. LAYOUT ---
layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("Snowfall Dashboard", className="fw-bold text-primary"),
                    html.P("Real-time seasonal tracking vs. 30-year Normals (1991-2020).", className="text-muted"),
                    html.Hr()
                ]
            )
        ),

        # Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Filter by State:", className="fw-bold"),
                        dcc.Dropdown(
                            id="state-filter",
                            options=[{'label': s, 'value': s} for s in available_states],
                            placeholder="Select a State...",
                            clearable=True
                        )
                    ],
                    width=12, md=4, className="mb-3"
                ),
                dbc.Col(
                    [
                        dbc.Alert(
                            "Click any station on the map to see its Seasonal Accumulation Report!",
                            color="info",
                            className="d-flex align-items-center"
                        )
                    ],
                    width=12, md=8
                )
            ]
        ),

        # Content
        dbc.Row(
            [
                # Map Column
                dbc.Col(
                    dcc.Graph(id="snow-map", style={"height": "600px"}),
                    width=12, lg=6, className="mb-4"
                ),
                
                # Graph Column
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Station Analysis (July - June)", className="fw-bold bg-light"),
                            dbc.CardBody(
                                dcc.Loading(
                                    dcc.Graph(id="station-graph", style={"height": "550px"}),
                                    type="circle"
                                )
                            )
                        ],
                        className="h-100 shadow-sm border-0"
                    ),
                    width=12, lg=6, className="mb-4"
                )
            ]
        )
    ],
    fluid=True,
    className="py-4"
)


# --- 4. CALLBACKS ---

@callback(
    Output("snow-map", "figure"),
    Input("state-filter", "value")
)
def update_map(selected_state):
    dff = df_map.copy()
    zoom = 3
    center = {"lat": 39.5, "lon": -98.35}

    if selected_state:
        dff = dff[dff['state'] == selected_state]
        if not dff.empty:
            center = {"lat": dff['latitude'].mean(), "lon": dff['longitude'].mean()}
            zoom = 6

    fig = px.scatter_mapbox(
        dff,
        lat="latitude", lon="longitude",
        color="annual_snow", size="annual_snow",
        hover_name="name",
        custom_data=["GHCN_ID", "state", "elevation"], 
        color_continuous_scale=px.colors.sequential.Plotly3,
        size_max=15, zoom=zoom, center=center,
        mapbox_style="carto-positron",
        title=f"Snowfall Normals ({selected_state if selected_state else 'US'})"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


@callback(
    Output("station-graph", "figure"),
    Input("snow-map", "clickData")
)
def update_graph(clickData):
    if not clickData:
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{"text": "Select a station...", "showarrow": False, "font": {"size": 20}}]
        )
        return fig

    # 1. Info from Click
    point = clickData['points'][0]
    station_id = point['customdata'][0]
    station_name = point['hovertext']

    # 2. Get & Process Normals (July -> June)
    stn_normals = df_normals[df_normals['GHCN_ID'] == station_id].copy()
    
    # Define Hydrological Year Order (July=7 ... June=6)
    month_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    month_names = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    
    # Create empty list for ordered normal values
    norm_vals = []
    for m in month_order:
        row = stn_normals[stn_normals['month'] == m]
        raw_val = row['MLY-SNOW-NORMAL'].values[0] if not row.empty else 0.0
        norm_vals.append(round(raw_val, 1)) # <--- ROUNDED
        
    # Calculate Accumulation for Normals
    norm_accum = np.round(np.cumsum(norm_vals), 1) # <--- ROUNDED

    # 3. Get & Process Current Data (Daily -> Monthly)
    df_daily = get_daily_acis_data(station_id)
    
    curr_vals = [0.0] * 12      # Monthly Sums
    curr_accum = [np.nan] * 12  # Monthly Accumulation
    missing_count = 0
    last_ob_date = "N/A"
    
    if df_daily is not None and not df_daily.empty:
        # A. Calc Missing Days
        missing_count = df_daily['snow'].isna().sum()
        
        # B. Find Last Observation
        valid_df = df_daily.dropna(subset=['snow'])
        if not valid_df.empty:
            last_ob_date = valid_df['date'].max().strftime("%Y-%m-%d")
        
        # C. Resample to Monthly
        monthly_sums = df_daily.groupby('month')['snow'].sum()
        
        running_total = 0
        has_started = False
        
        for i, m_num in enumerate(month_order):
            if m_num in monthly_sums.index:
                val = round(monthly_sums[m_num], 1) # <--- ROUNDED
                curr_vals[i] = val
                
                running_total += val
                curr_accum[i] = round(running_total, 1) # <--- ROUNDED
                
                has_started = True
            elif has_started:
                # Stop accumulation line if we are in future months
                curr_accum[i] = np.nan

    # 4. Build Dual-Axis Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # -- TRACE 1: Normal Monthly (Bars) --
    fig.add_trace(
        go.Bar(x=month_names, y=norm_vals, name='Normal Monthly', marker_color='lightgray', opacity=0.5),
        secondary_y=False
    )

    # -- TRACE 2: Current Monthly (Bars) --
    fig.add_trace(
        go.Bar(x=month_names, y=curr_vals, name='Current Monthly', marker_color='#0d6efd', opacity=0.8),
        secondary_y=False
    )
    
    # -- TRACE 3: Normal Accumulation (Line) --
    fig.add_trace(
        go.Scatter(x=month_names, y=norm_accum, name='Normal Accum.', mode='lines', 
                   line=dict(color='gray', width=2, dash='dot')),
        secondary_y=True
    )

    # -- TRACE 4: Current Accumulation (Line) --
    fig.add_trace(
        go.Scatter(x=month_names, y=curr_accum, name='Current Accum.', mode='lines+markers',
                   line=dict(color='#0d6efd', width=3)),
        secondary_y=True
    )

    # --- UPDATED LAYOUT ---
    title_text = f"<b>{station_name}</b><br><span style='font-size:14px;color:#555;'>Latest Obs: {last_ob_date} | Missing Days: {missing_count}</span>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Month",
        barmode='group',
        # LEGEND AT BOTTOM
        legend=dict(
            orientation="h", 
            y=-0.25,        
            x=0.5, 
            xanchor="center"
        ),
        template="simple_white",
        hovermode="x unified",
        margin=dict(b=80)   
    )
    
    fig.update_yaxes(title_text="Monthly Snow (in)", secondary_y=False)
    fig.update_yaxes(title_text="Accumulated Snow (in)", secondary_y=True, showgrid=False)

    return fig