import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/about')

layout = dbc.Container([
    dbc.Row([
        # Column 1: Image
        dbc.Col(
            html.Img(src="https://via.placeholder.com/400", className="img-fluid rounded-circle shadow p-2 border"),
            width=12, md=4, className="text-center mb-4"
        ),
        # Column 2: Bio
        dbc.Col([
            html.H1("Hi, I'm Becky Bolinger.", className="display-4 mb-3"),
            html.H4("Climate Scientist & Data Developer", className="text-muted mb-4"),
            html.P(
                """
                I specialize in translating complex climate data into actionable insights. 
                With a background in Python, data visualization, and climate science, 
                I build tools that help communities understand their changing environment.
                """,
                className="lead"
            ),
            html.Hr(),
            html.H5("Skills"),
            html.Div([
                dbc.Badge("Python", color="primary", className="me-2 p-2"),
                dbc.Badge("Dash / Plotly", color="primary", className="me-2 p-2"),
                dbc.Badge("Climate Data Analysis", color="info", className="me-2 p-2"),
                dbc.Badge("Data Visualization", color="success", className="me-2 p-2"),
            ])
        ], width=12, md=8)
    ], className="align-items-center mt-5")
])