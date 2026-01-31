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
                    html.Small("Select specific months to see seasonal totals/averages.", className="text-muted"),

                    html.Label("View Mode:", className="fw-bold mt-3 d-block"),
                    dbc.RadioItems(
                        id='view-mode',
                        options=[
                            {'label': 'Absolute Values', 'value': 'absolute'},
                            {'label': 'Departure from Avg', 'value': 'anomaly'},
                            {'label': 'Rankings', 'value': 'rank'}
                        ],
                        value='absolute',
                        inline=True,
                        className="mb-3"
                    ),

                    html.Label("Baseline Period (for Departure):", className="fw-bold"),
                    dbc.RadioItems(
                        id='baseline-mode',
                        options=[
                            {'label': '20th Century (1901-2000)', 'value': 'century'},
                            {'label': 'Climate Normals (1991-2020)', 'value': 'normal'}
                        ],
                        value='century', # Default per your request
                        inline=False,
                        className="mb-3"
                    ),
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

# calculate data rankings
def calculate_ncei_ranks(df, variable):
    """
    Adds 'Rank', 'Bin_ID', and 'Rank_Text' columns based on NCEI logic.
    Rank 1 = Coldest/Driest. Rank N = Warmest/Wettest.
    """
    # We cannot rank a NaN value, so we remove rows where Value is missing.
    df = df.dropna(subset=['Value']).copy()
    
    # 1. Calculate Rank (1 to N)
    # method='min' means ties get the same lower rank (e.g. two 1sts)
    df['Rank'] = df.groupby('State_Code')['Value'].rank(method='min', ascending=True)
    
    # 2. Get the Count (N) for each state to calculate percentiles
    df['Count'] = df.groupby('State_Code')['Value'].transform('count')
    
    # 3. Define the Terms based on Variable
    if 'Precipitation' in variable:
        low_adj, high_adj = "Driest", "Wettest"
    else:
        low_adj, high_adj = "Coolest", "Warmest"

    # 4. Define Helper for "1st", "2nd", "3rd"
    def ordinal(n):
        return "%d%s" % (n, {1:"st", 2:"nd", 3:"rd"}.get(n if n<20 else n%10, "th"))

    def get_rank_properties(row):
        rank = row['Rank']
        n = row['Count']
        
        # Calculate Percentile (0.0 to 1.0)
        pct = rank / n
        
        # --- NCEI BINNING LOGIC ---
        # Bins: 0=RecLow, 1=MuchBelow, 2=Below, 3=Near, 4=Above, 5=MuchAbove, 6=RecHigh
        
        # RECORD LOW (Rank 1)
        if rank == 1:
            bin_id = 0
            cat_text = f"Record {low_adj}"
            rank_text = f"1st {low_adj}"
            
        # RECORD HIGH (Rank N)
        elif rank == n:
            bin_id = 6
            cat_text = f"Record {high_adj}"
            rank_text = f"1st {high_adj}" # "1st Warmest"
            
        # MUCH BELOW (Bottom 10%)
        elif pct <= 0.10:
            bin_id = 1
            cat_text = "Much Below Average"
            rank_text = f"{ordinal(int(rank))} {low_adj}"
            
        # BELOW (Bottom 33%)
        elif pct <= 0.333:
            bin_id = 2
            cat_text = "Below Average"
            rank_text = f"{ordinal(int(rank))} {low_adj}"
            
        # NEAR AVERAGE (Middle 33%)
        elif pct <= 0.666:
            bin_id = 3
            cat_text = "Near Average"
            # For near average, we still label it based on which side of the median it is
            if pct <= 0.5:
                rank_text = f"{ordinal(int(rank))} {low_adj}"
            else:
                rank_text = f"{ordinal(int(n - rank + 1))} {high_adj}"

        # ABOVE (Top 33%)
        elif pct <= 0.90:
            bin_id = 4
            cat_text = "Above Average"
            rank_text = f"{ordinal(int(n - rank + 1))} {high_adj}"
            
        # MUCH ABOVE (Top 10%)
        else:
            bin_id = 5
            cat_text = "Much Above Average"
            rank_text = f"{ordinal(int(n - rank + 1))} {high_adj}"
            
        return pd.Series([bin_id, cat_text, rank_text])

    # Apply the logic row-by-row
    df[['Bin_ID', 'Category', 'Rank_Label']] = df.apply(get_rank_properties, axis=1)
    return df

# --- Helper Function for Aggregation ---
def get_aggregated_data(df_subset, variable, selected_months, view_mode='absolute', baseline_mode='century'):
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
            df_annual = df_annual[df_annual['Month_Count'] == expected_count].drop(columns=['Month_Count'])

    else:
        # Standard Annual Aggregation (No strict counting needed for simple annual)
        # (Unless you want to enforce 12 months for annual too? Usually safe to leave as is)
        is_precip = 'Precipitation' in variable 
        agg_func = 'sum' if is_precip else 'mean'
    
        df_annual = df_subset.groupby(['Year', 'State_Code', 'Variable'])['Value'].agg(agg_func).reset_index()
    
    # --- NEW: ANOMALY CALCULATION ---
    # Even if view_mode is 'absolute', we calculate these columns so they are available for Hover data
    
    # 1. Define Baseline Years
    if baseline_mode == 'century':
        start_year, end_year = 1901, 2000
    else: # 'normal'
        start_year, end_year = 1991, 2020
        
    # 2. Calculate LTA (Long Term Average) for each state
    # Filter for baseline years -> Group by State -> Calculate Mean
    baseline_df = df_annual[(df_annual['Year'] >= start_year) & (df_annual['Year'] <= end_year)]
    lta_series = baseline_df.groupby('State_Code')['Value'].mean()

    # 3. Map LTA back to the main dataframe
    df_annual['LTA'] = df_annual['State_Code'].map(lta_series)
    
    # 4. Calculate Anomaly
    df_annual['Anomaly'] = df_annual['Value'] - df_annual['LTA']
    
    # 5. Handle View Mode
    # We save the "Absolute" value in a specific column for safe keeping
    df_annual['Absolute_Value'] = df_annual['Value']
    
    if view_mode == 'anomaly':
        # Swap 'Value' to be the Anomaly so the Map plots the anomaly automatically
        df_annual['Value'] = df_annual['Anomaly']

    elif view_mode == 'rank':
        # If Ranking, we calculate the ranks and populate metadata
        df_annual = calculate_ncei_ranks(df_annual, variable)
        # Note: We keep 'Value' as the Absolute Value for the CHART, 
        # but the MAP will use 'Bin_ID' for coloring.

    return df_annual

# --- Callback: Update Map ---
@app.callback(
    [Output('us-map', 'figure'),
     Output('map-title', 'children'),
     Output('status-text', 'children')],
    [Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value'),
     Input('view-mode', 'value'),
     Input('baseline-mode', 'value')]
)
def update_map(selected_variable, selected_months, view_mode, baseline_mode):
    # --- PHASE 1: SETUP & UNITS ---
    if 'Precipitation' in selected_variable:
        units = 'Inches'
        standard_colors = "Viridis"
        anomaly_colors = "BrBG"  # Brown(Dry) -> Green(Wet)
        # NCEI Precipitation Colors (Record Dry -> Record Wet)
        rank_colors = ["#543005", "#8C510A", "#BF812D", "#F7F7F7", "#80CDC1", "#35978F", "#003C30"]
    else:
        units = '°F'
        standard_colors = "RdBu_r" # Blue(Cold) -> Red(Hot)
        anomaly_colors = "RdBu_r"
        # NCEI Temperature Colors (Record Cold -> Record Warm)
        rank_colors = ["#053061", "#2166AC", "#4393C3", "#F7F7F7", "#F4A582", "#D6604D", "#67001F"]

    # Filter Global Data by Variable first
    dff = df[df['Variable'] == selected_variable]

    # --- PHASE 2: DETERMINE TIME SELECTION ---
    if selected_months and 'annual' in selected_months:
        target_months = ['annual']
        time_label = "Annual"
        
    elif not selected_months:
        target_months = [latest_date.month]
        # FIX: Force just the month name (e.g. "December") to avoid double years
        time_label = latest_date.strftime('%B') 
        
    else:
        target_months = [m for m in selected_months if isinstance(m, int)]
        time_label = format_title_months(target_months)
        
    # --- PHASE 3: GET AGGREGATED DATA ---
    # Now we make just ONE call to the helper function
    dff_aggregated = get_aggregated_data(dff, selected_variable, target_months, view_mode, baseline_mode)
    
    # Get the single latest year for the map
    latest_year = dff_aggregated['Year'].max()
    dff_map = dff_aggregated[dff_aggregated['Year'] == latest_year]

    # --- PHASE 4: CONFIGURE VIEW MODE (Visuals) ---
    baseline_text = "1901-2000" if baseline_mode == 'century' else "1991-2020"
    
    # OPTION 1: RANKINGS
    if view_mode == 'rank':
        color_col = 'Bin_ID'       # 0-6 bins
        colorscale = rank_colors   # Specific NCEI Hex codes
        range_color = [0, 6]       # Force distinct bins
        midpoint = None
        
        map_title = f"{selected_variable} - {time_label} {latest_year} (Rankings)"
        status_msg = f"Showing rankings for {time_label} {latest_year}."
        
        # Hover shows Category ("Much Above Average") and Rank ("3rd Driest")
        hover_data = {'Category': True, 'Rank_Label': True, 'Absolute_Value': ':.2f'}
        labels_map = {'Category': 'Class', 'Rank_Label': 'Rank', 'Absolute_Value': 'Value'}
        
    # OPTION 2: DEPARTURE (ANOMALY)
    elif view_mode == 'anomaly':
        color_col = 'Value'        # This column now holds the Anomaly (calculated in helper)
        colorscale = anomaly_colors
        range_color = None
        midpoint = 0               # Force White to be 0
        
        map_title = f"{selected_variable} - {time_label} {latest_year} (Departure from {baseline_text})"
        status_msg = f"Showing departure from {baseline_text} average."
        
        hover_data = {'LTA': ':.2f', 'Absolute_Value': ':.2f'}
        labels_map = {'Value': f'Departure ({units})', 'LTA': 'Long Term Avg', 'Absolute_Value': 'Actual'}
        
    # OPTION 3: ABSOLUTE (STANDARD)
    else:
        color_col = 'Value'        # This column holds the standard value
        colorscale = standard_colors
        range_color = None
        midpoint = None
        
        map_title = f"{selected_variable} - {time_label} {latest_year}"
        status_msg = f"Showing absolute values for {time_label} {latest_year}."
        
        hover_data = {'LTA': ':.2f'}
        labels_map = {'Value': units, 'LTA': 'Long Term Avg'}

    # --- PHASE 5: DRAW MAP ---
    fig = px.choropleth(
        dff_map,
        locations='State_Code',
        locationmode="USA-states",
        color=color_col,
        scope="usa",
        color_continuous_scale=colorscale,
        range_color=range_color,
        color_continuous_midpoint=midpoint,
        hover_name="State_Code",
        hover_data=hover_data,
        labels=labels_map
    )

    # Hide the legend for rankings (optional, cleaner look)
    if view_mode == 'rank':
        fig.update_layout(coloraxis_showscale=False)
        
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig, map_title, status_msg

# --- Callback: Update Chart ---
@app.callback(
    Output('trend-chart', 'figure'),
    [Input('us-map', 'clickData'),
     Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value'),
     Input('view-mode', 'value'),
     Input('baseline-mode', 'value')]
)
def update_chart(clickData, selected_variable, selected_months, view_mode, baseline_mode):
    # --- PHASE 1: SETUP & UNITS ---
    if 'Precipitation' in selected_variable:
        units = 'Inches'
        anomaly_colors = "BrBG"
        # NCEI Precipitation Colors
        rank_colors = ["#543005", "#8C510A", "#BF812D", "#F7F7F7", "#80CDC1", "#35978F", "#003C30"]
    else:
        units = '°F'
        anomaly_colors = "RdBu_r"
        # NCEI Temperature Colors
        rank_colors = ["#053061", "#2166AC", "#4393C3", "#F7F7F7", "#F4A582", "#D6604D", "#67001F"]

    # Determine State
    if clickData is None:
        state_code = 'CO'
    else:
        state_code = clickData['points'][0]['location']
        
    dff = df[(df['State_Code'] == state_code) & (df['Variable'] == selected_variable)]

    # --- PHASE 2: TIME SELECTION ---
    if selected_months and 'annual' in selected_months:
        target_months = ['annual']
        time_label = "Annual"
    elif not selected_months:
        target_months = [latest_date.month]
        time_label = latest_date.strftime('%B') # Fix: Just Month Name
    else:
        target_months = [m for m in selected_months if isinstance(m, int)]
        time_label = format_title_months(target_months)

    # --- PHASE 3: AGGREGATION ---
    dff_plot = get_aggregated_data(dff, selected_variable, target_months, view_mode, baseline_mode)

    # --- PHASE 4: CONFIGURE CHART ---
    baseline_text = "1901-2000" if baseline_mode == 'century' else "1991-2020"
    
    # OPTION 1: RANKINGS (Now a Bar Chart!)
    if view_mode == 'rank':
        chart_type = 'bar'
        y_col = 'Absolute_Value' 
        y_label = units
        
        # We color by Bin_ID to match the map (Dark Red/Blue bars for extreme years)
        color_col = 'Bin_ID'
        colorscale = rank_colors
        range_color = [0, 6]
        midpoint = None
        
        title_text = f"{state_code} - {selected_variable} History (Rankings)"
        hover_template = f"<b>Year:</b> %{{x}}<br><b>Value:</b> %{{y}} {units}<br><b>Rank:</b> %{{customdata[2]}}"
    
    # OPTION 2: DEPARTURE (ANOMALY)
    elif view_mode == 'anomaly':
        chart_type = 'bar'
        y_col = 'Value' # Anomaly
        y_label = f"Anomaly ({units})"
        
        color_col = 'Value'
        colorscale = anomaly_colors
        range_color = None
        midpoint = 0
        
        title_text = f"{state_code} - {time_label} Departure from {baseline_text} Average"
        hover_template = f"<b>Year:</b> %{{x}}<br><b>Anomaly:</b> %{{y}} {units}"
        
    # OPTION 3: ABSOLUTE (STANDARD)
    else:
        chart_type = 'line'
        y_col = 'Value'
        y_label = units
        color_col = None # Single color line
        
        title_text = f"{state_code} - {selected_variable} Trend ({time_label})"
        hover_template = f"<b>Year:</b> %{{x}}<br><b>Value:</b> %{{y}} {units}"

    # --- PHASE 5: DRAW CHART ---
    if chart_type == 'bar':
        fig = px.bar(
            dff_plot, 
            x='Year', 
            y=y_col, 
            title=title_text,
            color=color_col, 
            color_continuous_scale=colorscale,
            range_color=range_color,
            color_continuous_midpoint=midpoint,
            labels={'Value': y_label, 'Year': ''}
        )
        
        # If Ranking, we need to pass the custom text for hover
        if view_mode == 'rank':
             fig.update_traces(customdata=dff_plot[['Bin_ID', 'Category', 'Rank_Label']])
             fig.update_traces(hovertemplate=hover_template)
             # Hide the color bar for rankings (0-6 is confusing on a chart legend)
             fig.update_layout(coloraxis_showscale=False)
        else:
            # For Anomalies, we SHOW the legend (color bar)
            fig.update_layout(coloraxis_showscale=True)
            
        # Add baseline 0 line
        if view_mode == 'anomaly':
            fig.add_hline(y=0, line_width=2, line_color="black")
            
    else: # Line Chart (Standard View)
        fig = px.line(
            dff_plot, 
            x='Year', 
            y=y_col, 
            title=title_text,
            markers=True,
            labels={'Value': y_label, 'Year': ''}
        )

    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_title=y_label
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)