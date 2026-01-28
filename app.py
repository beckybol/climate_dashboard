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

# 3. Define the Layout using Rows and Columns
app.layout = dbc.Container([
    
    # --- ROW 1: The Navbar ---
    dbc.NavbarSimple(
        brand="NOAA Climate Explorer",
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
                dbc.CardHeader("Filter Settings"),
                dbc.CardBody([
                    html.Label("Select Variable:", className="fw-bold"),
                    dcc.Dropdown(
                        id='variable-dropdown',
                        options=[{'label': i, 'value': i} for i in df['Variable'].unique()],
                        value='Average Temp',
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
                            {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}
                        ],
                        value=[],
                        multi=True,
                        placeholder="Select months..."
                    ),
                    html.Small("Leave empty to see annual trends.", className="text-muted")
                ])
            ], className="shadow-sm") # Adds a subtle shadow
        ], width=12, md=4), # Takes 12 columns on phone, 4 on desktop

        # COLUMN 2: The Map (Width = 8/12 columns)
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='us-map', clickData=None, style={'height': '400px'})
                ], style={'padding': '0'}) # Remove padding so map fits tight
            ], className="shadow-sm")
        ], width=12, md=8)
    ], className="mb-4"),

    # --- ROW 3: The Trend Chart ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Historical Trend Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='trend-chart')
                ])
            ], className="shadow-sm")
        ], width=12)
    ])

], fluid=True) # fluid=True uses the full width of the screen

# 4. Define Callbacks (Logic remains the same!)
@app.callback(
    Output('us-map', 'figure'),
    Input('variable-dropdown', 'value')
)
def update_map(selected_variable):
    dff = map_df[map_df['Variable'] == selected_variable]
    fig = px.choropleth(
        dff,
        locations='State_Code',
        locationmode="USA-states",
        color='Value',
        scope="usa",
        color_continuous_scale="Viridis" if selected_variable == 'Precipitation' else "RdBu_r"
    )
    # Update layout to remove margins for a cleaner look in the card
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
    Output('trend-chart', 'figure'),
    [Input('us-map', 'clickData'),
     Input('variable-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_chart(clickData, selected_variable, selected_months):
    if clickData is None:
        state_code = 'CO' 
    else:
        state_code = clickData['points'][0]['location']
        
    dff = df[df['State_Code'] == state_code]
    dff = dff[dff['Variable'] == selected_variable]
    
    title_suffix = ""
    if selected_months:
        dff = dff[dff['Month'].isin(selected_months)]
        title_suffix = f" (Selected Months)"
    
    dff = dff.sort_values('Date')
    
    fig = px.line(
        dff, 
        x='Date', 
        y='Value', 
        title=f"Time Series for {state_code}: {selected_variable}{title_suffix}"
    )
    fig.update_layout(template="simple_white") # Clean white background for chart
    return fig

# 5. Run
if __name__ == '__main__':
    app.run(debug=True)