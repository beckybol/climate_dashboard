import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

dash.register_page(__name__, title='Blog | Climate Becky', image='assets/images/snow_forecast.png')

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
    {
        "id": "noreaster-feb-2026",  # <--- NEW: Unique ID for linking
        "title": "Epic Nor'easter Brings Record Snowfall to the Northeast",
        "date": "February 23, 2026",
        "tags": ["Snowfall", "Winter Storm"],
        "content": [
            {"image": "assets/images/northeast_satellite.jpg", "caption": "GOES-19 snapshot of the winter storm on the morning of February 23, 2026. Satellite imagery courtesy of NOAA and CSU/CIRA."},
            {"text": "A major winter storm, named Winter Storm Hernando by the Weather Channel, has blanketed much of the Northeast with heavy snow and strong winds. While nor'easters are common in this region, this storm is likely to go down in the record books once the final totals are tallied."},
            {"text": "The Community Collaborative Rain, Hail, and Snow Network (CoCoRaHS) is a fantastic resource for mapping a high density of snowfall observations for this event. As of Monday morning (February 23), snow totals over 1 foot extended from Delaware to Maine. 24-hour totals ranged between 15 and 25 inches for parts of New Jersey, New York, Connecticut, and Rhode Island. Weekly totals (which capture the bulk of the event through Monday morning) have exceed 2 feet in some locations."},
            {"image": "assets/images/cocorahs_snowfall.png", "caption": "Accumulated snowfall observed over northeast from February 16 - February 23, 2026. Map courtesy of CoCoRaHS."},
            {"text": "Snow continued to accumulate through Monday, and additional snowfall is expected through Tuesday. While totals aren't final yet, confirmation of a broken record has already been reported in Rhode Island. As of 1pm local time, the airport in Providence, RI had received 32.8\" of snow, breaking the record of 28.6\" set during the Blizzard of 1978."},
            {"text": "New York City has seen its fair share from this storm as well. On top of an already snowy winter, Central Park has received over 30 inches for the season, beating out 9 of the last 10 years."},
            {"image": "assets/images/nyc_accum.png", "caption": "Daily snowfall accumulation at Central Park COOP station since November 2025 through February 23, 2026. Average accumulation shown in brown line, and most recent 10 seasons also plotted. Data and graph courtesy of ACIS."},
            {"text": "The persistent weather pattern that has locked the western U.S. in a ridge for much of the winter has also contributed to the cold and snow experienced in the northeast. The first day of climatological spring (March 1) will bring a close to the winter season, with the west finishing warmer than average and the east colder than average. That pattern will probably be obvious in the seasonal snowfall totals as well."},
            {"text": "Check out my new snowfall dashboard, which shows monthly snowfall totals and accumulations for this season compared to average. For example, Loveland (my city) typically sees about 32\" of snow by the end of February, with 8\" of that just from February alone. This season, we've gotten a whopping 10.7 inches total. By comparison, Atlantic City, NJ usually gets about 17 inches annually, and they're already over 16 inches. Search for your area to see how you're doing compared to average!"},
            {"button": "Launch Snowfall Dashboard", "link": "/snow_dashboard"},
            {"image": "assets/images/loveland_snow.png", "caption": "Monthly snowfall totals for Loveland, CO for the 2025-2026 season compared to average. Data and graph courtesy of ACIS."},     
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
        elif "button" in block:
            post_children.append(
                html.Div(
                    dbc.Button(
                        block["button"],            # The text on the button
                        href=block["link"],         # Where it goes
                        color="primary",            # Bootstrap color
                        className="px-4 py-2 text-uppercase fw-bold" # Styling
                    ),
                    className="text-center mb-4"    # Centers the button in the post
                )
            )

    # Add Share Buttons at the bottom of the content list
    post_children.append(make_share_buttons(post["title"], post["id"]))
    
    return dbc.Card(
        dbc.CardBody(post_children),
        id=post["id"],  # <--- IMPORTANT: This assigns the ID to the HTML element
        className="mb-5 border-0"
    )

# --- 4. HELPER: SOCIAL SHARE BUTTONS ---
def make_share_buttons(post_title, post_id):
    # 1. Define the base URL of your site (CHANGE THIS to your actual domain!)
    base_url = "https://climatebecky.com/blog"
    
    # 2. Build the specific link to this post
    post_link = f"{base_url}#{post_id}"
    
    # 3. Create the social links
    # These are special URLs that open the platform's sharing tool
    linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={post_link}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={post_link}"
    bluesky_text = f"{post_title}: {post_link}"
    bluesky_url = f"https://bsky.app/intent/compose?text={bluesky_text}"

# --- ICONS ---
    # 1. Bluesky (Loaded from file)
    bluesky_icon = html.Img(
        src="assets/bluesky.svg", 
        style={"height": "18px", "width": "18px"}
    )

    # 2. LinkedIn & Facebook (Standard Bootstrap Icons)
    linkedin_icon = html.I(className="bi bi-linkedin", style={"fontSize": "18px", "color": "#0a66c2"})
    facebook_icon = html.I(className="bi bi-facebook", style={"fontSize": "18px", "color": "#1877f2"})

    # --- BUTTON STYLE ---
    # Flexbox centers the icon perfectly inside the button
    btn_style = {
        "width": "32px", 
        "height": "32px", 
        "display": "flex", 
        "alignItems": "center", 
        "justifyContent": "center",
        "border": "none",
        "background": "transparent"
    }

    return html.Div(
        [
            html.Small("Share: ", className="text-muted me-2"),
            
            # LinkedIn
            dbc.Button(linkedin_icon, href=linkedin_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
            
            # Bluesky
            dbc.Button(bluesky_icon, href=bluesky_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
            
            # Facebook
            dbc.Button(facebook_icon, href=facebook_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
        ],
        className="d-flex align-items-center mt-3"
    )

# --- 5. LAYOUT ---
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