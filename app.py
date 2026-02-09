import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the app with the 'pages' plugin
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.LUX]) 
# 'LUX' is a clean, professional theme. Try 'FLATLY' or 'MATERIA' if you prefer.

server = app.server

# --- THE NAVBAR ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Climate Becky", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Climate Dashboard", href="/cag_dashboard", active="exact")),
        dbc.NavItem(dbc.NavLink("About Me", href="/about", active="exact")),
        dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio", active="exact")),
        dbc.NavItem(dbc.NavLink("Connect", href="/contact", active="exact")),
    ],
    brand="Becky Bolinger",
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-4",
)

# --- THE FOOTER ---
footer = html.Div(
    [
        html.Hr(),
        html.P("© 2026 Becky Bolinger | Built with Python Dash & Plotly", className="text-center text-muted"),
    ],
    className="mt-5 p-4 bg-light"
)

# --- MAIN LAYOUT ---
app.layout = html.Div([
    navbar,
    dbc.Container([
        dash.page_container  # This is where the specific page content loads!
    ], fluid=True, className="px-4"), # fluid=True uses full width, nice for dashboards
    footer
])

if __name__ == "__main__":
    app.run(debug=True)