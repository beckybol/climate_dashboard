import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the app with the 'pages' plugin
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP]) 
# 'LUX' is a clean, professional theme. Try 'FLATLY' or 'MATERIA' if you prefer.


server = app.server

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/logo.png", height="100px")),
                        #dbc.Col(dbc.NavbarBrand("Climate Becky", className="ms-2 fw-bold text-dark")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            
            dbc.Collapse(
                dbc.Nav(
                    [
                        # Added 'text-dark' to links just to be safe, though light=True handles most
                        dbc.NavItem(dbc.NavLink("Climate Becky", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/cag_dashboard", active="exact")),
                        dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio", active="exact")),
                        dbc.NavItem(dbc.NavLink("About Me", href="/about", active="exact")),
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="#F7F0FA",    # <--- Changed from 'dark'
    dark=False,        # <--- This ensures text is dark colored
    className="shadow sticky-top py-1", # <--- Adds a nice shadow and keeps nav visible while scrolling
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
    
    # We wrap the content in a Div with a background color
    html.Div([
        dbc.Container([
            dash.page_container
        ], fluid=True, className="px-4 py-4") # Added padding
    ], style={"backgroundColor": "#f0f2f5", "minHeight": "100vh"}), # <--- The "Facebook Gray" color
    
    footer
])

# --- CALLBACK FOR MOBILE MENU ---
@dash.callback(
    dash.Output("navbar-collapse", "is_open"),
    [dash.Input("navbar-toggler", "n_clicks")],
    [dash.State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run(debug=True)