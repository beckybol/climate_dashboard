import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# 1. Initialize the App
app = dash.Dash(__name__)
server = app.server

# 2. Load the Data
df = pd.read_csv('master_climate_data.csv')

# Pre-calculate a default map dataset (Average of all time for each state)
# This makes the initial map render faster than processing the whole CSV every time
map_df = df.groupby(['State_Code', 'State', 'Variable'])['Value'].mean().reset_index()

# 3. Define the Layout (The "HTML" Structure)
app.layout = html.Div([
    
    # Header
    html.H1("US Climate Data Explorer", style={'textAlign': 'center'}),
    
    # Control Panel (Flexbox for side-by-side dropdowns)
    html.Div([
        # Variable Dropdown
        html.Div([
            html.Label("Select Variable:"),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[{'label': i, 'value': i} for i in df['Variable'].unique()],
                value='Average Temp',  # Default value
                clearable=False
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Month Dropdown (Multi-select)
        html.Div([
            html.Label("Filter by Month (Optional):"),
            dcc.Dropdown(
                id='month-dropdown',
                options=[
                    {'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                    {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                    {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                    {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                    {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                    {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}
                ],
                value=[],    # Default is empty (means "All Months")
                multi=True,  # Allows selecting multiple months (e.g., "March-May")
                placeholder="Select months to filter seasonality..."
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'padding': '20px'}),

    # Map Section
    html.Div([
        dcc.Graph(id='us-map', clickData=None) # Start with no click data
    ]),

    # Trend Chart Section
    html.Div([
        dcc.Graph(id='trend-chart')
    ], style={'padding': '20px'})
])

# 4. Define Callbacks (The Logic)

# Callback A: Update the Map when the Variable changes
@app.callback(
    Output('us-map', 'figure'),
    Input('variable-dropdown', 'value')
)
def update_map(selected_variable):
    # Filter our pre-calculated map data for the selected variable
    dff = map_df[map_df['Variable'] == selected_variable]
    
    fig = px.choropleth(
        dff,
        locations='State_Code',      # The column with 'CO', 'TX', etc.
        locationmode="USA-states",
        color='Value',               # The data to visualize
        scope="usa",
        title=f"Long-term Average: {selected_variable}",
        color_continuous_scale="Viridis" if selected_variable == 'Precipitation' else "RdBu_r"
    )
    return fig

# Callback B: Update the Trend Chart when Map is clicked OR Inputs change
@app.callback(
    Output('trend-chart', 'figure'),
    [Input('us-map', 'clickData'),
     Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_chart(clickData, selected_variable, selected_months):
    # Default to Colorado if nothing is clicked yet
    if clickData is None:
        state_code = 'CO' 
    else:
        state_code = clickData['points'][0]['location']
        
    # 1. Filter by State
    dff = df[df['State_Code'] == state_code]
    
    # 2. Filter by Variable
    dff = dff[dff['Variable'] == selected_variable]
    
    # 3. Filter by Month (if any are selected)
    title_suffix = ""
    if selected_months:
        dff = dff[dff['Month'].isin(selected_months)]
        title_suffix = f" (Selected Months Only)"
    
    # 4. Sort by date to ensure the line connects correctly
    dff = dff.sort_values('Date')
    
    # 5. Build the Chart
    fig = px.line(
        dff, 
        x='Date', 
        y='Value', 
        title=f"Time Series for {state_code}: {selected_variable}{title_suffix}"
    )
    
    return fig

# 5. Run the Server
if __name__ == '__main__':
    app.run(debug=True)