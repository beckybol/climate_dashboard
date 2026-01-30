import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc  # <--- NEW IMPORT
import plotly.express as px
import pandas as pd

# 1. Initialize the App with a Professional Theme
# external_stylesheets pulls in the "Flatly" theme (looks like a modern SaaS app)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# 2. Load the Data
df = pd.read_csv('master_climate_data.csv')
map_df = df.groupby(['State_Code', 'State', 'Variable'])['Value'].mean().reset_index()

#Ensure Date is datetime object for sorting/filtering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

MONTH_MAP = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December',
    'annual': 'Annual'
}

# FIND THE LATEST DATE for the default map view
latest_date = df['Date'].max()
latest_month_name = MONTH_MAP[latest_date.month] + " " + str(latest_date.year)

def format_title_months(month_list):
    if not month_list:
        return "Annual"
    
    # 1. Detect Winter Wrap-Around (Nov/Dec + Jan/Feb)
    # Check if we have both late months (Nov/Dec) and early months (Jan/Feb)
    has_late = any(m >= 11 for m in month_list)
    has_early = any(m <= 2 for m in month_list)
    
    if has_late and has_early:
        # Custom Sorting for Title: Put Nov(11) and Dec(12) BEFORE Jan(1)
        # We treat 11 as -1 and 12 as 0 for sorting purposes
        def winter_sort(m):
            return m if m < 7 else m - 13
            
        m_sorted = sorted(month_list, key=winter_sort)
        # Now [1, 11, 12] becomes [11, 12, 1]
    else:
        # Standard Sort
        m_sorted = sorted(month_list)
    
    # Check if consecutive
    # If [11, 12, 1], the winter_sort values are [-2, -1, 1]. 
    # Not perfectly linear math, so simpler to just check if it looks like a range.
    
    first_name = MONTH_MAP[m_sorted[0]][:3] # "Nov"
    last_name = MONTH_MAP[m_sorted[-1]][:3] # "Jan"
    
    # If we have more than 2 months and they span a range
    if len(m_sorted) > 2:
        return f"{first_name} - {last_name}"
    
    # Otherwise list them
    return ", ".join([MONTH_MAP[m][:3] for m in m_sorted])

# 3. Define the Layout using Rows and Columns
app.layout = dbc.Container([
    
    # --- ROW 1: The Navbar ---
    dbc.NavbarSimple(
        brand="NOAA Climate at a Glance Explorer",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"  # Adds margin at the bottom
    ),

    # --- ROW 2: The Main Content Area ---
    dbc.Row([
        
        # COLUMN 1: The Control Panel (Width = 4/12 columns)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Configuration"),
                dbc.CardBody([
                    html.Label("Select Variable:", className="fw-bold"),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': i, 'value': i} for i in df['Variable'].unique()],
                        value='Average Temperature',
                        clearable=False,
                        className="mb-3" # Margin bottom
                    ),
                    
                    html.Label("Filter by Season/Month:", className="fw-bold"),
                    dcc.Dropdown(
                        id='month-dropdown',
                        options=[
                            {'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                            {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                            {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                            {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                            {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                            {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}, 
                            {'label': 'Annual (All Months)', 'value': 'annual'}
                        ],
                        multi=True,
                        placeholder="Select months..."
                    ),
                    html.Small("Select specific months to see seasonal totals/averages.", className="text-muted")
                ])
            ], className="shadow-sm mb-4"), # Adds a subtle shadow
    
            # Add a small instruction card or stats summary here if needed
            dbc.Card([
                dbc.CardBody([
                    html.H5("Current View", className="card-title"),
                    html.P(id='status-text', className="card-text")
                ])
            ], className="shadow-sm")
            
        ], width=12, md=4), # Takes 12 columns on phone, 4 on desktop

        # Map Area
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(id='map-title', children="US Map"),
                dbc.CardBody([
                    dcc.Graph(id='us-map', clickData=None, style={'height': '450px'})
                ], style={'padding': '0'})
            ], className="shadow-sm")
        ], width=12, md=8)
    ], className="mb-4"),

    # --- ROW 3: The Trend Chart ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Historical Time Series"),
                dbc.CardBody([
                    dcc.Graph(id='trend-chart')
                ])
            ], className="shadow-sm")
        ], width=12)
    ])

], fluid=True) # fluid=True uses the full width of the screen

# 4. Callbacks

# --- Helper Function for Aggregation ---
def get_aggregated_data(df_subset, variable, selected_months):
    # 1. Filter by Month (if not Annual)
    if selected_months and 'annual' not in selected_months:
        months_list = [m for m in selected_months if isinstance(m, int)]
        
        if months_list:
            # We must know how many months we EXPECT (e.g., 3 for Dec-Jan-Feb)
            expected_count = len(months_list)
            
            df_subset = df_subset[df_subset['Month'].isin(months_list)].copy()
            
            # --- INTELLIGENT SEASON LOGIC ---
            min_m = min(months_list)
            max_m = max(months_list)
            count = len(months_list)
            
            # Check Contiguity
            is_continuous = (max_m - min_m) == (count - 1)
            
            # Apply Shift if NOT continuous (implies wrap-around like Dec-Jan)
            if not is_continuous:
                df_subset['Year'] = df_subset.apply(
                    lambda x: x['Year'] + 1 if x['Month'] >= 7 else x['Year'], axis=1
                )

            # --- AGGREGATION WITH STRICT COUNTING ---
            is_precip = 'Precipitation' in variable 
            agg_func = 'sum' if is_precip else 'mean'
            
            # Group by Year/State and calculate BOTH the Value and the Count of months
            df_annual = df_subset.groupby(['Year', 'State_Code', 'Variable']).agg(
                Value=('Value', agg_func),
                Month_Count=('Month', 'count')
            ).reset_index()
            
            # CRITICAL FILTER: Drop years that are missing months
            # If we asked for 3 months, but 2026 only has 1 (Dec), drop 2026.
            df_annual = df_annual[df_annual['Month_Count'] == expected_count]
            
            # Clean up (remove the count column)
            return df_annual.drop(columns=['Month_Count'])

    # Standard Annual Aggregation (No strict counting needed for simple annual)
    # (Unless you want to enforce 12 months for annual too? Usually safe to leave as is)
    is_precip = 'Precipitation' in variable 
    agg_func = 'sum' if is_precip else 'mean'
    
    df_annual = df_subset.groupby(['Year', 'State_Code', 'Variable'])['Value'].agg(agg_func).reset_index()
    
    return df_annual

# --- Callback: Update Map ---
@app.callback(
    [Output('us-map', 'figure'),
     Output('map-title', 'children'),
     Output('status-text', 'children')],
    [Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_map(selected_variable, selected_months):
    # 1. Determine Units
    if 'Precipitation' in selected_variable:
        units = 'Inches'
    else:
        units = '°F'

    # Filter Global Data
    dff = df[df['Variable'] == selected_variable]

    # Handling the "Map View"
    # CASE 1: ANNUAL Selected -> Show Aggregated Annual Data
    if selected_months and 'annual' in selected_months:
        # Calculate full annual averages/sums
        dff_aggregated = get_aggregated_data(dff, selected_variable, ['annual'])
        
        # Find latest available year in the aggregated data
        latest_agg_year = dff_aggregated['Year'].max()
        dff_map = dff_aggregated[dff_aggregated['Year'] == latest_agg_year]
        
        map_title = f"{selected_variable} - {latest_agg_year} Annual Average"
        status_msg = f"Showing Annual Average for {latest_agg_year}."

# CASE 2: EMPTY Selection -> Show Latest Available Month (Raw Data)
    elif not selected_months:
        dff_map = dff[dff['Date'] == latest_date]
        
        map_title = f"{selected_variable} - {latest_month_name}"
        status_msg = f"Showing latest data: {latest_month_name}."
    
    else:
        # Complex Case: Seasonal Aggregation
        # STEP 1: Aggregate the ENTIRE history first (allows the "Year Shift" to work)
        dff_aggregated = get_aggregated_data(dff, selected_variable, selected_months)
        
        # STEP 2: Find the latest available "Season Year"
        latest_season_year = dff_aggregated['Year'].max()
        
        # STEP 3: Filter the AGGREGATED data for that year
        dff_map = dff_aggregated[dff_aggregated['Year'] == latest_season_year]
        
        # Title Formatting
        # Use our helper to format [1, 11, 12] -> "Nov - Jan"
        month_names_str = format_title_months([m for m in selected_months if isinstance(m, int)])
        
        map_title = f"{selected_variable} - {month_names_str} {latest_season_year}"
        status_msg = f"Showing {month_names_str} average for Season {latest_season_year}."

    fig = px.choropleth(
        dff_map,
        locations='State_Code',
        locationmode="USA-states",
        color='Value',
        scope="usa",
        color_continuous_scale="Viridis" if 'Precipitation' in selected_variable else "RdBu_r",
        hover_name="State_Code",
        labels={'Value': units}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig, map_title, status_msg

# --- Callback: Update Chart ---
@app.callback(
    Output('trend-chart', 'figure'),
    [Input('us-map', 'clickData'),
     Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_chart(clickData, selected_variable, selected_months):
    if 'Precipitation' in selected_variable:
        units = "Precipitation (inches)"
    else:
        units = "Temperature (°F)"

    # 1. Determine State
    if clickData is None:
        state_code = 'CO' # Default
    else:
        state_code = clickData['points'][0]['location']
        
    # 2. Filter Master Data by State & Variable
    dff = df[(df['State_Code'] == state_code) & (df['Variable'] == selected_variable)]
    
# CASE 1: ANNUAL Selected -> Show Annual Trend
    if selected_months and 'annual' in selected_months:
        dff_plot = get_aggregated_data(dff, selected_variable, ['annual'])
        title_text = f"{state_code} - {selected_variable} (Annual Trend)"
    
    # CASE 2: EMPTY Selection -> Show "Latest Month" Trend (e.g. History of Decembers)
    elif not selected_months:
        # Filter for the month of the latest_date global variable
        target_month = latest_date.month
        target_month_name = latest_date.strftime('%B')
        
        dff_plot = dff[dff['Month'] == target_month].sort_values('Year')
        title_text = f"{state_code} - {selected_variable} ({target_month_name} Trend)"

 # CASE 3: SEASONAL Selection -> Show Seasonal Trend
    else:
        dff_plot = get_aggregated_data(dff, selected_variable, selected_months)
        
        month_names_str = format_title_months([m for m in selected_months if isinstance(m, int)])
        title_text = f"{state_code} - {selected_variable} ({month_names_str} Trend)"

    # 5. Plot
    fig = px.line(
        dff_plot, 
        x='Year', 
        y='Value', 
        title=title_text,
        markers=True, # Adds dots to the line so you can see the individual yearly points
        labels={'Value': units, 'Year': 'Year'}
    )
    
    # Optional: Add OLS Trendline if you want
    # fig = px.scatter(..., trendline="ols", ...) 
    
    fig.update_layout(template="simple_white")
    return fig

if __name__ == '__main__':
    app.run(debug=True)