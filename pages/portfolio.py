import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/portfolio')

# --- PROJECT CARD HELPER ---
def create_project_card(title, description, link, image_url="https://via.placeholder.com/300x200"):
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardImg(src=image_url, top=True),
                dbc.CardBody(
                    [
                        html.H4(title, className="card-title"),
                        html.P(description, className="card-text"),
                        dbc.Button("View Project", color="primary", href=link, external_link=True),
                    ]
                ),
            ],
            className="h-100 shadow-sm hover-shadow", # h-100 makes them equal height
        ),
        width=12, md=6, lg=4, className="mb-4" # Responsive grid (1 col on phone, 3 on desktop)
    )

layout = dbc.Container([
    html.H2("My Portfolio", className="mb-4 display-5 fw-bold"),
    html.P("A collection of my work in climate data analysis, visualization, and web development.", className="lead mb-5"),
    
    dbc.Row([
        # Project 1
        create_project_card(
            "US Climate Dashboard", 
            "An interactive dashboard visualizing temperature and precipitation anomalies using NOAA NCEI data.",
            "/"
        ),
        # Project 2 (Example)
        create_project_card(
            "Colorado Drought Report", 
            "Analysis of the 2018 Four Corners drought using CoCoRaHS reports and stakeholder impact data.",
            "#",
            image_url="https://via.placeholder.com/300x200?text=Drought+Report"
        ),
        # Project 3 (Example)
        create_project_card(
            "Streamflow Regression", 
            "Comparing linear regression models to estimate missing gage data for water resource planning.",
            "#",
            image_url="https://via.placeholder.com/300x200?text=Streamflow+Analysis"
        ),
    ])
])