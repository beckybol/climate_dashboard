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

# FIND THE LATEST DATE for the default map view
latest_date = df['Date'].max()
latest_month_name = latest_date.strftime('%B %Y')

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
                        value='annual',
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
    """
    Groups data by Year.
    - If Precipitation: SUMs the months.
    - If Temperature: AVERAGES the months.
    """
    # 1. Filter by Month (if not Annual)
    if selected_months and 'annual' not in selected_months:
        # If user selects specific months (e.g., [6, 7, 8])
        # Ensure selected_months is a list of integers
        months_list = [m for m in selected_months if isinstance(m, int)]
        if months_list:
            df_subset = df_subset[df_subset['Month'].isin(months_list)]
    
    # 2. Define Aggregation Method based on Variable
    # Assuming 'Precipitation' is the exact string in your CSV
    is_precip = 'Precipitation' in variable 
    agg_func = 'sum' if is_precip else 'mean'
    
    # 3. Group by Year and Aggregate
    # We group by ['Year', 'State_Code', 'Variable'] to keep those columns available
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
    # LOGIC:
    # 1. If "Annual" or nothing selected -> Show LATEST AVAILABLE MONTH (as requested)
    # 2. If Specific Months selected -> Show the Aggregated Average/Sum for the LATEST YEAR in the dataset?
    #    OR show the Long-Term Average for that season?
    #    Standard practice for "Interactive Maps" is to show the *most recent* valid data point 
    #    that matches the filter, or an average. 
    #    Let's stick to your request: "Initial loading... latest month". 
    
    # Assign units based on variable
    if 'Precipitation' in selected_variable:
        units = "inches"
    else:
        units = "°F"

    # Default trigger: Use Latest Date
    target_date = latest_date
    map_title = f"{selected_variable} - {latest_month_name}"
    status_msg = f"Showing data for {latest_month_name}."

    # Filter Global Data
    dff = df[df['Variable'] == selected_variable]

    # Handling the "Map View"
    # If the user is in "Annual" mode, we show the LATEST MONTH (December 2025)
    if not selected_months or 'annual' in selected_months:
        dff_map = dff[dff['Date'] == latest_date]
    
    else:
        # If user selects specific months (e.g. JJA), what should the map show?
        # Option A: The JJA average for the LAST available year (2025).
        # This matches the "Latest" philosophy.
        
        # 1. Filter for selected months
        months_list = [m for m in selected_months if isinstance(m, int)]
        dff_season = dff[dff['Month'].isin(months_list)]
        
        # 2. Get the latest year available in this filtered set
        latest_year = dff_season['Year'].max()
        
        # 3. Aggregate for that specific year
        # (e.g. Average Temp for JJA 2025)
        dff_map = get_aggregated_data(dff_season[dff_season['Year'] == latest_year], selected_variable, months_list)
        
        # Update Titles
        month_names = ", ".join([str(m) for m in months_list]) # You could map 1->Jan here if desired
        map_title = f"{selected_variable} - {latest_year} (Months: {month_names})"
        status_msg = f"Map showing aggregation for Year {latest_year}. Chart below shows trend."

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
    
    # 3. Run Aggregation Logic (The "One Value Per Year" Rule)
    # This function handles the Sum vs Mean logic and Month filtering
    dff_aggregated = get_aggregated_data(dff, selected_variable, selected_months)
    
    # 4. Generate Title
    if not selected_months or 'annual' in selected_months:
        subtitle = "Annual"
        if 'Precipitation' in selected_variable:
            subtitle += " Total"
        else:
            subtitle += " Average"
    else:
        subtitle = "Seasonal"
        if 'Precipitation' in selected_variable:
            subtitle += " Total"
        else:
            subtitle += " Average"

    # 5. Plot
    fig = px.line(
        dff_aggregated, 
        x='Year', 
        y='Value', 
        title=f"{state_code} - {selected_variable} ({subtitle})",
        markers=True, # Adds dots to the line so you can see the individual yearly points
        labels={'Value': units, 'Year': 'Year'}
    )
    
    # Optional: Add OLS Trendline if you want
    # fig = px.scatter(..., trendline="ols", ...) 
    
    fig.update_layout(template="simple_white")
    return fig

if __name__ == '__main__':
    app.run(debug=True)