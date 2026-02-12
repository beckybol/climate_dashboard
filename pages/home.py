import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', title='Home | Climate Becky')

# --- SECTION 1: HERO ---
hero_section = dbc.Container(
    [
        html.H1("Turning Climate Data into Action.", className="display-3 fw-bold text-primary"),
        html.P(
            "I'm Becky Bolinger. I build interactive tools that make complex climate science accessible, understandable, and usable.",
            className="lead text-muted mb-4",
            style={"maxWidth": "700px"}
        ),
        dbc.Stack(
            [
                dbc.Button("View My Portfolio", href="/portfolio", color="primary", size="lg", className="me-3"),
                dbc.Button("Read the Blog", href="/blog", color="outline-dark", size="lg"),
            ],
            direction="horizontal",
        )
    ],
    fluid=True,
    className="py-5"
)

# --- SECTION 2: FEATURED DASHBOARD ---
featured_dashboard = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.CardImg(
                        src="assets/dashboard-preview.png", # You'll need to take a screenshot!
                        className="img-fluid rounded-start",
                        style={"height": "100%", "objectFit": "cover"}
                    ),
                    className="col-md-6",
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H6("FEATURED PROJECT", className="text-uppercase text-primary fw-bold mb-2"),
                            html.H2("US Climate Anomalies Dashboard", className="card-title"),
                            html.P(
                                "An interactive exploration of temperature and precipitation anomalies across the US. "
                                "Built with Python Dash and NOAA NCEI data to visualize extreme weather patterns.",
                                className="card-text text-muted",
                            ),
                            dbc.Button("Launch Dashboard", href="/cag_dashboard", color="dark", className="mt-3"),
                        ]
                    ),
                    className="col-md-6 d-flex align-items-center",
                ),
            ],
            className="g-0 d-flex",
        )
    ],
    className="mb-5 shadow-sm border-0 overflow-hidden",
)

# --- SECTION 3: LATEST UPDATES (BLOG & SOCIALS) ---
updates_section = dbc.Row(
    [
        # Left Column: Blog Teasers
        dbc.Col(
            [
                html.H3("Latest from the Blog", className="mb-4"),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.H5("Migrating to M3: A Python Nightmare", className="mb-1"),
                                        html.Small("Feb 12, 2026", className="text-muted"),
                                    ],
                                    className="d-flex w-100 justify-content-between",
                                ),
                                html.P("How I fixed my Conda environment and got Dash running on Apple Silicon.", className="mb-1 text-muted small"),
                            ],
                            action=True,
                            href="/blog/python-m3-migration", # We'll build this link later
                            className="border-0 border-bottom p-3"
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.H5("Visualizing Drought in Colorado", className="mb-1"),
                                        html.Small("Jan 20, 2026", className="text-muted"),
                                    ],
                                    className="d-flex w-100 justify-content-between",
                                ),
                                html.P("Using CoCoRaHS reports to tell the story of the 2018 Four Corners drought.", className="mb-1 text-muted small"),
                            ],
                            action=True,
                            href="/blog/colorado-drought",
                            className="border-0 border-bottom p-3"
                        ),
                    ],
                    flush=True,
                ),
                dbc.Button("View All Posts", href="/blog", color="link", className="mt-2 text-decoration-none p-0"),
            ],
            md=7, className="pe-md-5 mb-5"
        ),
        
        # Right Column: Social "Connect"
        dbc.Col(
            [
                html.H3("Let's Connect", className="mb-4"),
                html.P("I share daily updates on climate science and data viz.", className="text-muted mb-4"),
                
                dbc.Stack(
                    [
                        dbc.Button(
                            [html.I(className="bi bi-linkedin me-2"), " LinkedIn"],
                            href="https://linkedin.com/in/climatebecky",
                            color="primary",
                            outline=True,
                            className="w-100 text-start mb-2"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-github me-2"), " GitHub"],
                            href="https://github.com/climatebecky",
                            color="dark",
                            outline=True,
                            className="w-100 text-start mb-2"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-bluesky me-2"), " Bluesky"],
                            href="https://bsky.app/profile/climatebecky.com",
                            color="dark",
                            outline=True,
                            className="w-100 text-start"
                        ),
                    ],
                    gap=2
                )
            ],
            md=5,
            className="bg-light p-4 rounded-3"
        )
    ]
)

# --- FINAL LAYOUT ASSEMBLY ---
layout = dbc.Container(
    [
        hero_section,
        html.Hr(className="my-5 opacity-25"),
        featured_dashboard,
        html.Hr(className="my-5 opacity-25"),
        updates_section,
    ],
    fluid=False, # Standard container for readability
    className="py-4"
)