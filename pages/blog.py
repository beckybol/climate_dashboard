import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

dash.register_page(__name__, title='Blog | Climate Becky')

# --- 1. DEFINE POSTS (THE BLOCK SYSTEM) ---
# Each item in "content" can be:
# - {"text": "Your paragraph..."}
# - {"image": "assets/filename.png", "caption": "Optional caption"}
# - {"graph": {data...}}

# --- 1. DATA WITH ANCHOR IDs ---
blog_posts = [
    {
        "id": "snowpack-feb-2026",  # <--- NEW: Unique ID for linking
        "title": "Seasonal Snowpack Update",
        "date": "February 16, 2026",
        "tags": ["Snowfall", "Snowpack", "Water Supply"],
        "content": [
            {"text": "A multi-day storm system is dropping big totals across the western U.S. this week. The Sierra Nevadas are likely to get over 3 feet of new snow, while the Interior Rockies and Cascades will see between 1-2 feet."},
            {"image": "assets/images/snow_forecast.png", "caption": "72-hour snowfall forecast for the western U.S., as of February 16, 2026. Map from the Weather Prediction Center."},
            {"text": "This storm is a welcome relief for many drought-stricken areas, especially given the near record low snowpack so far this season. NRCS snowpack is below to well-below average for most of the west. A startling number of SNOTEL sites in Colorado and Utah are reporting levels below the 5th percentile (meaning they are drier than 95% of historical records for this date)."},
            {"image": "assets/images/swe_map.jpeg", "caption": "Current snowpack conditions across the western U.S. as of February 16, 2026. Map from NRCS SNOTEL."},
            {"text": "At the Colorado Headwaters, snowpack is record low for this time of year. With only 56 days to go till normal peak snowpack, new accumulations would have to be record-breaking to get the basin to near-average levels. A much below average peak and early melt are much more likely. This has major implications for water supply, agriculture, and wildfire risk across the region."},
            {"image": "assets/images/co_swe_time.png", "caption": "Projected peak snowpack for the Colorado Headwaters, as of February 16, 2026. Data from NRCS."},
            {"text": "These concerningly low snowpack accumulations are evident in the Colorado Basin River Forecast Center's water supply forecasts. The April-July runoff forecast for Lake Powell (the Upper Colorado River Basin's largest reservoir) is 38% of average, about a 4-million acre-foot deficit."},
            {"image": "assets/images/powell_inflows.png", "caption": "April-July runoff forecast for Lake Powell. Official forecast values from February 1, 2026. Data from the Colorado Basin River Forecast Center."},
            {"text": "While we can expect snowpack and forecasted runoff to improve with the current storm, the overall outlook for the season remains concerning. Additional late season snows and colder temperatures could minimize further deteriorating conditions and the risk of large wildfires this summer. Stay tuned over the next couple of months!"},
        ]
    },
]

# --- 2. HELPER: BUILD THE SIDEBAR ---
def make_sidebar():
    return html.Div(
        [
            html.H5("Recent Posts", className="fw-bold text-primary mb-3"),
            dbc.Nav(
                [
                    dbc.NavLink(
                        post["title"],
                        href=f"#{post['id']}",  # Links to the specific ID on this page
                        external_link=True,     # Forces the browser to handle the jump
                        className="text-muted ps-0 py-1"
                    )
                    for post in blog_posts
                ],
                vertical=True,
                className="mb-5"
            ),
            
            html.H5("Popular Tags", className="fw-bold text-primary mb-3"),
            html.Div(
                [
                    # These are just visual for now, but they show your topics!
                    dbc.Badge("Snowfall", color="light", text_color="dark", className="me-1 mb-1"),
                    dbc.Badge("Snowpack", color="light", text_color="dark", className="me-1 mb-1"),
                    dbc.Badge("Water Supply", color="light", text_color="dark", className="me-1 mb-1"),
                ]
            )
        ],
        className="sticky-top pt-5", # Keeps sidebar visible while scrolling!
        style={"top": "80px", "zIndex": "1"} # Adjust 'top' to clear your navbar
    )

# --- 3. HELPER: BUILD POSTS ---
def make_climate_post(post):
    post_children = []
    
    # Header
    post_children.append(html.Div(
        [
            html.Small(post["date"], className="text-muted text-uppercase fw-bold"),
            html.H3(post["title"], className="card-title fw-bold text-primary mb-2"),
            html.Div(
                [dbc.Badge(t, color="light", text_color="primary", className="me-1") for t in post["tags"]],
                className="mb-4"
            )
        ]
    ))

    # Content Blocks
    for block in post["content"]:
        if "text" in block:
            post_children.append(html.P(block["text"], className="card-text mb-3"))
        elif "image" in block:
            img_component = html.Img(src=block["image"], className="img-fluid rounded border shadow-sm w-100")
            if "caption" in block:
                post_children.append(html.Figure([img_component, html.Figcaption(block["caption"], className="figure-caption text-center mt-1")], className="mb-4"))
            else:
                post_children.append(html.Div(img_component, className="mb-4"))
        elif "graph" in block:
            fig = px.bar(x=block["graph"]["x"], y=block["graph"]["y"], title=block["graph"]["title"])
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            post_children.append(dcc.Graph(figure=fig, className="shadow-sm border rounded mb-4"))

    return dbc.Card(
        dbc.CardBody(post_children),
        id=post["id"],  # <--- IMPORTANT: This assigns the ID to the HTML element
        className="mb-5 border-0"
    )

# --- 4. LAYOUT ---
layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left Column: The Blog Feed (Width 8)
                dbc.Col(
                    [
                        html.H1("Becky's Blog", className="display-6 fw-bold text-primary mb-4"),
                        html.P("Latest updates on the state of the climate, drought, extremes, and anomalies.", className="lead text-muted"),
                        html.Hr(className="my-5"),
                        html.Div([make_climate_post(post) for post in blog_posts])
                    ],
                    width=12, lg=8
                ),
                
                # Right Column: The Sidebar (Width 4)
                dbc.Col(
                    make_sidebar(),
                    width=12, lg=4,
                    className="d-none d-lg-block" # Hides sidebar on small mobile screens
                )
            ]
        )
    ],
    className="py-5"
)