import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import os
import plotly.express as px

dash.register_page(__name__, title='Blog | Climate Becky', image='assets/images/snow_forecast.png')

# --- 1. DEFINE POSTS (THE BLOCK SYSTEM) ---
# Each item in "content" can be:
# - {"text": "Your paragraph..."}
# - {"image": "assets/filename.png", "caption": "Optional caption"}
# - {"graph": {data...}}

current_dir = os.path.dirname(os.path.abspath(__file__))
records_csv = os.path.join(current_dir, '..', 'data', 'snow_records_202602.csv')

try:
    df_records = pd.read_csv(records_csv)
    
    # 2. Build the interactive map
    fig_records = px.scatter_mapbox(
        df_records,
        lat="latitude",
        lon="longitude",
        color="snowfall",
        size="snowfall",
        hover_name="Station Name",
        hover_data={
            "latitude": False, 
            "longitude": False,
            "snowfall": True,
            "years in record": True, 
            "Station Type": True
        },
        color_continuous_scale=px.colors.diverging.Portland, # A nice snow-themed color scale
        size_max=20,
        zoom=5,
        center={"lat": 40.5, "lon": -73.0}, # Centers roughly over New England (based on your data)
        mapbox_style="carto-positron",
        title="Record Tied/Broken Storm Totals (Feb 2026)"
    )
    fig_records.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

except FileNotFoundError:
    # Fallback if the file is missing
    fig_records = px.scatter(title="Data File Not Found")

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
        "date": "February 24, 2026",
        "tags": ["Snowfall", "Winter Storm"],
        "content": [
            {"image": "assets/images/northeast_satellite.jpg", "caption": "GOES-19 snapshot of the winter storm on the morning of February 23, 2026. Satellite imagery courtesy of NOAA and CSU/CIRA."},
            {"text": "A major winter storm, named Hernando by the Weather Channel, blanketed much of the Northeast with heavy snow and strong winds. While nor'easters are common in the region, this storm is likely to go down in the record books once the final totals are official."},
            {"text": "Storm totals for the 48 hours preceding Tuesday monring (February 24), show widespread areas along the northeast coast from Delaware to Maine receiving over a foot of snow. Forty observers reported over 30 inches of snow."},
            {"image": "assets/images/ne_snow_totals.png", "caption": "Accumulated snowfall reports over northeast from February 22 - February 24, 2026. Map courtesy of the National Weather Service."},
            {"text": "Official reports of a broken record was reported in Rhode Island by the National Weather Service (NWS). By Monday evening, the airport in Providence, RI had received over 37 inches of snow, breaking the record of 28.6\" set during the Blizzard of 1978. Quite the feat, considering the station's long history of snowfall records dating back to 1932. In the map below, explore the other storm total records that were tied or broken, 21 in total."},
            {"figure": fig_records},
            {"text": "New York City has seen its fair share from this storm as well. A NWS COOP station near Islip on Long Island has now seen its snowiest February on record (reporting since 1963). On top of an already snowy winter, Central Park has received 42 inches for the season, beating out the last 10 years."},
            {"image": "assets/images/islip_monthly_snow.png", "caption": "February monthly snowfall totals over time at Islip, NY. Data and graph courtesy of ACIS."},
            {"image": "assets/images/nyc_accum.png", "caption": "Daily snowfall accumulation at Central Park COOP station since November 2025 through February 23, 2026 (green shaded). Average accumulation shown in brown line, and most recent 10 seasons also plotted. Data and graph courtesy of ACIS."},
            {"text": "The persistent weather pattern that has locked the western U.S. in a ridge for much of the winter has also contributed to the active weather pattern in the northeast. The first day of climatological spring (March 1) will bring a close to the winter season, with the west finishing warmer than average and the east colder than average. That pattern will be evident in the seasonal snowfall totals as well."},
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
        elif "figure" in block:
            post_children.append(
                dcc.Graph(
                    figure=block["figure"], 
                    className="shadow-sm border rounded mb-4",
                    config={"scrollZoom": True} # Lets users zoom the map with their mouse!
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
        dcc.Location(id="blog-url", refresh=False),
        html.Div(id="scroll-dummy"),        

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
    fluid=True,
    className="py-5"
)

# --- AUTO-SCROLL CALLBACK ---
# This forces the browser to scroll to the #id after the page finishes loading
dash.clientside_callback(
    """
    function(url) {
        // Check if there is a #hash in the URL
        if (window.location.hash) {
            // Wait 500 milliseconds for Dash to render the blog posts
            setTimeout(function() {
                // Find the element with that ID (removing the '#' symbol)
                var element = document.getElementById(window.location.hash.substring(1));
                if (element) {
                    // Scroll it into view smoothly
                    element.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            }, 500); 
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("scroll-dummy", "children"), # We just need a dummy output
    Input("blog-url", "href")           # Triggers whenever the URL changes
)