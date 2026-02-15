import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc

dash.register_page(__name__, title='Portfolio | Climate Becky', path='/portfolio')

# --- DATA: YOUR PROJECTS ---
# This list drives the whole page. Easy to edit!
projects = [
    {
        "id": "climate-report",
        "title": "Climate Change in Colorado Report",
        "desc": "A comprehensive synthesis of climate science for water resources management. I led the development, stakeholder engagement, and website creation for this major state report.",
        "link": "https://climatechange.colostate.edu", # Direct link (no pop-up needed)
        "tags": ["Report", "Website", "Project Lead"]
    },
    {
        "id": "ccc-website",
        "title": "Colorado Climate Center Website",
        "desc": "I led the first major redesign of the center's website in 20 years, organizing complex climate data into a user-friendly, interactive portal for the state.",
        "link": "https://climate.colostate.edu",
        "tags": ["Web Design", "Data Viz", "Climate Communication"]
    },
    {
        "id": "rcc-acis",
        "title": "RCC ACIS Web Services",
        "desc": "Powerful tools that transform raw ACIS data into value-added products like temperature graphs, climate extremes trackers, and custom data access tools.",
        "tags": ["Python", "API", "JavaScript"],
        # This one HAS sub-items, so it will trigger a Pop-Up
        "details": [
            {
                "name": "Data Access Portal",
                "desc": "Interactive map and dropdowns for fetching raw station metadata and data.",
                "link": "https://climate.colostate.edu/data_access_new.html"
            },
            {
                "name": "Daily Temperature Graphs",
                "desc": "Interactive graphs comparing current temps to normals, records, and averages.",
                "link": "https://climate.colostate.edu/temp_graph.html"
            },
            {
                "name": "Climate Extremes Tracker",
                "desc": "Statewide and station-specific climate extremes generated on the fly.",
                "link": "https://climate.colostate.edu/extremes.html"
            }
        ]
    },
    {
        "id": "coagmet",
        "title": "CoAgMET Tools",
        "desc": "A suite of interactive products for Colorado's agricultural mesonet, including real-time maps, wind roses, and soil moisture monitors.",
        "tags": ["Agriculture", "Real-time Data", "Mapping"],
        "details": [
            {"name": "Current Conditions Map", "desc": "Real-time temp, wind, and precip for 95+ stations.", "link": "https://coagmet.colostate.edu"},
            {"name": "Station Pages", "desc": "Dynamic dashboards for individual station data and metadata.", "link": "https://coagmet.colostate.edu/stn_page.php?avn01"},
            {"name": "Wind Roses", "desc": "Interactive wind summaries calculated in Python from historical records.", "link": "https://coagmet.colostate.edu/wind_summaries.php"},
            {"name": "Soil Moisture Monitor", "desc": "Interactive tracking of soil conditions at multiple depths.", "link": "https://coagmet.colostate.edu/soil_monitor.php"}
        ]
    },
    {
        "id": "ncei-normals",
        "title": "NOAA NCEI Climate Normals",
        "desc": "Visualization tools making the massive NOAA normals database accessible, including frost dates, degree days, and precipitation thresholds.",
        "tags": ["NOAA Data", "Climate Normals", "Data Viz"],
        "details": [
            {"name": "Normals Maps", "desc": "Interactive map of 1991-2020 normals for all CO stations.", "link": "https://climate.colostate.edu/normals_stn_select.html"},
            {"name": "Station Graphs", "desc": "Comprehensive charts of temp/precip normals and thresholds.", "link": "https://climate.colostate.edu/station_normal.html?USW00003017"},
            {"name": "Gridded Normals", "desc": "Static maps generated from the ACIS Gridded Normals Mapper.", "link": "https://climate.colostate.edu/normals_maps.html"}
        ]
    },
    {
        "id": "cag",
        "title": "NOAA Climate at a Glance",
        "desc": "Custom implementations of NOAA's global analysis tool, focusing on Colorado-specific anomalies and agricultural crop insurance applications.",
        "tags": ["Anomalies", "Trends", "Analysis"],
        "details": [
            {"name": "Colorado Time Series", "desc": "Temperature and precipitation ranks and anomalies for CO climate divisions.", "link": "https://climate.colostate.edu/co_cag/cag_time.html"},
            {"name": "County Precip Tool", "desc": "Risk assessment tool for agricultural users and crop insurance.", "link": "https://climate.colostate.edu/county_precip_tool.html"}
        ]
    },
    {
        "id": "drought",
        "title": "Drought Monitoring Products",
        "desc": "Automated dashboards and SPI/SPEI maps designed to assist the state's weekly drought monitoring and decision-making process.",
        "tags": ["Drought", "Automation", "Operational"],
        "details": [
            {"name": "CO Drought Dashboard", "desc": "Daily/weekly updated dashboard of all key drought indicators.", "link": "https://climate.colostate.edu/drought"},
            {"name": "Interactive SPI Maps", "desc": "Maps of Standardized Precipitation Index matching US Drought Monitor scales.", "link": "https://climate.colostate.edu/spi_map.html"}
        ]
    },
]


# --- HELPER FUNCTIONS ---

def make_project_card(index, project):
    """Generates a Card for a single project."""
    
    # Decide what the button does
    if "details" in project:
        # It has sub-items -> Open Modal
        button = dbc.Button(
            "View Examples", 
            id={"type": "open-modal-btn", "index": index}, # Pattern Matching ID
            color="primary", 
            outline=True, 
            className="w-100"
        )
    else:
        # It's a direct link -> Go to URL
        button = dbc.Button(
            "Visit Website", 
            href=project.get("link", "#"), 
            target="_blank",
            color="dark", 
            outline=True, 
            className="w-100"
        )

    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [dbc.Badge(tag, color="light", text_color="dark", className="me-1 mb-2") for tag in project.get("tags", [])],
                    ),
                    html.H4(project["title"], className="card-title fw-bold"),
                    html.P(project["desc"], className="card-text text-muted small flex-grow-1"),
                    html.Div(button, className="mt-auto") # mt-auto pushes button to bottom
                ],
                className="d-flex flex-column h-100"
            ),
            className="h-100 shadow-sm border-0 hover-shadow"
        ),
        md=6, lg=4, className="mb-4" # 3 columns on large screens, 2 on medium
    )

# --- PAGE LAYOUT ---

layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("Portfolio", className="display-4 fw-bold text-primary mb-3"),
                    html.P(
                        "A collection of climate tools, dashboards, and reports I've developed to make science actionable.",
                        className="lead text-muted mb-5"
                    ),
                ],
                width=12
            )
        ),

        # Grid of Project Cards
        dbc.Row(
            [make_project_card(i, p) for i, p in enumerate(projects)],
            className="g-4 mb-5"
        ),

        # --- THE SHARED MODAL (Pop-Up) ---
        # This is hidden until a button is clicked
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
                dbc.ModalBody(id="modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal-btn", className="ms-auto", n_clicks=0)
                ),
            ],
            id="portfolio-modal",
            size="lg",    # Large pop-up
            centered=True, # Center on screen
            is_open=False,
            scrollable=True
        ),
    ],
    fluid=False,
    className="py-5"
)


# --- CALLBACKS ---

@callback(
    Output("portfolio-modal", "is_open"),
    Output("modal-title", "children"),
    Output("modal-body", "children"),
    [
        Input({"type": "open-modal-btn", "index": ALL}, "n_clicks"), # Listen to ALL "View Example" buttons
        Input("close-modal-btn", "n_clicks")
    ],
    [State("portfolio-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_click, is_open):
    """
    Handles opening the modal with specific content, or closing it.
    """
    trigger_id = ctx.triggered_id
    
    # Case 1: Close button clicked
    if trigger_id == "close-modal-btn":
        return False, "", ""

    # Case 2: One of the "View Examples" buttons was clicked
    # 'trigger_id' will look like: {'index': 2, 'type': 'open-modal-btn'}
    if trigger_id and trigger_id.get("type") == "open-modal-btn":
        
        # Get the index of the project that was clicked
        project_idx = trigger_id["index"]
        selected_project = projects[project_idx]
        
        # Build the content for the pop-up
        title = selected_project["title"]
        
        # Create a nice list of the sub-items
        content_list = dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.H5(item["name"], className="mb-1 text-primary"),
                                html.P(item["desc"], className="mb-1 text-muted small"),
                            ],
                            className="w-100"
                        ),
                        html.Small(html.I(className="bi bi-box-arrow-up-right"), className="text-muted")
                    ],
                    action=True,
                    href=item["link"],
                    target="_blank",
                    className="d-flex justify-content-between align-items-center"
                )
                for item in selected_project.get("details", [])
            ],
            flush=True
        )
        
        return True, title, content_list

    # Default: Do nothing
    return is_open, "", ""