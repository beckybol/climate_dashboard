import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize the app with the 'pages' plugin
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP]) 
# 'LUX' is a clean, professional theme. Try 'FLATLY' or 'MATERIA' if you prefer.


server = app.server

# --- THE NAVBAR ---
navbar = dbc.Navbar(
    dbc.Container(
        [
            # --- BRAND (Logo + Name) ---
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/logo.png", height="40px")), # Adjust height as needed
                        dbc.Col(dbc.NavbarBrand("Climate Becky", className="ms-2")),
                    ],
                    align="center",
                    className="g-0", # g-0 removes default grid gutters for tighter spacing
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            
            # --- HAMBURGER MENU (Mobile) ---
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            
            # --- NAV LINKS ---
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/cag_dashboard", active="exact")),
                        dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio", active="exact")),
                        dbc.NavItem(dbc.NavLink("About Me", href="/about", active="exact")),
                    ],
                    className="ms-auto", # ms-auto pushes links to the right
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="dark",
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