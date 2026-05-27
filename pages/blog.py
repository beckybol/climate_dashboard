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
        "id": "extreme-mar-2026",  # <--- NEW: Unique ID for linking
        "title": "A Deep Dive into the Extreme Heat in March 2026",
        "date": "April 17, 2026",
        "tags": ["Extreme Heat", "Climate Change", "Climate Variability"],
        "content": [
            {"text": "NOAA NCEI released its updated Climate at a Glance data for March on April 8, and the numbers are startling. March 2026 was the hottest March on record for 10 states, spanning from Oklahoma to California. For 12 states, this was the warmest October - March on record. The anomalous warmth for most of the last 6 months has been driven by a persistent ridge of high pressure that has been parked over the western U.S. This same pattern has locked Alaska in a deep trough, resulting in a cold winter and the state's fourth coldest March on record."},
            {"image": "assets/images/temp_ranks_20260409.png", "caption": "Average statewide temperature ranks for March 2026. Data from NOAA NCEI Climate at a Glance."},
            {"button": "Explore statewide anomalies and ranks", "link": "/cag_dashboard"},
            {"text": "Records in and of themselves are one thing, and it's not a surprise to see state temperature records broken as the climate continues to warm. But these March anomalies didn't just break records, they shattered them. For Colorado, Nevada, Arizona, Utah, and New Mexico, this was not only their warmest March on record, the monthly average would also place in the top 10 warmest Aprils on record. For California, this March was warmer than all Marches and all Aprils in the 132-year record. The below graphic shows just how extreme March 2026 temperatures were for California, compared to all Marches in the 132-year record."},
            {"image": "assets/images/ca_temp_dist.png", "caption": "March temperature distribution for California. Data from NOAA NCEI Climate at a Glance."},
            {"text": "Just how extreme was March 2026 compared to other Marches? Consider California's distribution of March temperatures. When compared to the historic record (pre-1980), this March was so warm, it would be a 1 in 1,000,000 year event. Even when comparing to the modern record (1980-2025), this March is still a 1-in-11,000 year event. While climate change has increased the likelihood of an extremely warm March, this one was still extremely unlikely to occur."},
            {"text": "For Arizona, March 2026 was a 1-in-200,000 year event in the modern record. Nine of the top 10 hottest March days in Phoenix occured in 2026 (records dating back to 1933) - this included 9 days over 100°F! The quadrant plot below shows just how much of an outlier March 2026 was compared to previous Marches for Arizona. For comparison, Idaho also recorded its warmest March, but the anomaly was still within the bounds of the distribution of previous Marches."},
            {"image": "assets/images/AZ_ID_quadrants_202603.png", "caption": "Quadrant plots showing March precipitation vs. temperature for Arizona (left) and Idaho (right). Data from NOAA NCEI Climate at a Glance."},
            {"text": "Looking at all months out of the year, a monthy anomaly greater than +10°F (over the 20th century average)is extremely rare. In fact, for California, Arizona, and New Mexico, it had never happened before. March 2026 surpassed +10°F in all three states. For Colorado and Utah, they both exceeded +10°F anomalies for the first time in December 2025. They since broke those records in March. See the table below for more stats. The last column shows the return period for these anomalies in the modern era. Both Arizona and New Mexico experienced an event that would occur less than 1 in 100,000 years; California and Utah's anomalies were over 1 in 10,000 year events; and for Colorado and Nevada, this event would be expected to occur less than 1 in every 5,000 years."},
            {"table": {
                "headers": ["State", "March 2026 Temperature (°F)", "Anomaly (°F)", "Modern Era Return Period"],
                "rows": [
                    ["Arizona", "63.4°F", "+13.9°F", "1 in 230,971 years"],
                    ["California", "61.4°F", "+12.6°F", "1 in 11,566 years"],
                    ["Colorado", "46.8°F", "+13.1°F", "1 in 6,289 years"],
                    ["Idaho", "41.4°F", "+8.8°F", "1 in 84 years"],
                    ["Nevada", "52.4°F", "+12.9°F", "1 in 5,483 years"],
                    ["New Mexico", "55.6°F", "+12.1°F", "1 in 359,013 years"],
                    ["Oklahoma", "60.5°F", "+11.2°F", "1 in 350 years"],
                    ["Texas", "66.8°F", "+10.5°F", "1 in 650 years"],
                    ["Utah", "51.1°F", "+13.8°F", "1 in 25,466 years"],
                    ["Wyoming", "41.7°F", "+12.6°F", "1 in 375 years"]
                ]
            }},
            {"text": "Portions of article originally published on LinkedIn."}
        ]
    },
    {
        "id": "heat-gdd-mar-2026",  # <--- NEW: Unique ID for linking
        "title": "It's Not Just Extreme Heat, It's Accumulated Warmth",
        "date": "April 1, 2026",
        "tags": ["Growing Degree Days", "Extreme Heat"],
        "content": [
            {"text": "It's hard to overstate how significant the warmth has been so far this year in Colorado. Seeing all of our trees green up in the last few weeks has me thinking about accumulated warmth. Normally we look at Growing Degree Days after April 1 (A GDD is basically the number of degrees warmer than 50°F for a day). But I wanted to see how GDD has accumulated for this year compared to other years. The result for Fort Collins was quite shocking. 2026 has left all other years in the dust."},
            {"image": "assets/images/gdd_fort_collins.png", "caption": "Total Growing Degree Days for Fort Collins, CO for January 1 - March 31 for all years, 1893-2026. Data and graph from ACIS."},
            {"text": "The above graphic shows total accumulated GDD for Jan 1 - Mar 31 for all years in the record. It should be noted that Fort Collins is a clear example of urban heat island, and that is evident in the trend over time. Still, it's not normal to get over 50 GDDs by April. 2012 was the extreme example, when GDD exceeded 100 before the start of April. This year? Through March 31 the GDD accumulation is 221."},
            {"image": "assets/images/gdd_accumulation_20260330.png", "caption": "Accumulated Growing Degree Days for Fort Collins, CO for 2026 (till the end of March) compared to average and record low years (2012) through May 31. Data and graph from ACIS."},
            {"text": "Having lived through the spring of 2012 and the horrible drought that followed, it's just mind boggling that 2026 has not just surpassed 2012, but clobbered that record. In 2012, GDD didn't reach 221 until April 24. And what's normal? Well, in a normal year we don't pass 221 GDD until May 19. Here's one more stat to shake you - even if April was cold and we didn't accumulated any more GDDs, it would still be the second highest GDD by April 30 out of the 130+ year record (and over 100 GDDs higher than average)."},
            {"text": "The outlook for April indicates it's likely to be above average temperatures, which means the GDD accumulation will continue to outpace the average. While a longer growing season may sound nice, there are some serious implications for agriculture, water demand, and wildfire risk. Early growth and blooms will be more vulnerable to late season freezes. There is already an increase in demand for irrigation water, but water restrictions are being put into place amidst severe drought across the state. The warmth also extends wildfire season, which is at risk to be a very active season this year. Stay tuned for more updates on the spring and summer outlook!"},
            {"text": "Portions of article originally published on LinkedIn."}
        ]
    },
    {
        "id": "snowpack-mar-2026",  # <--- NEW: Unique ID for linking
        "title": "Record Low Snowpack Across the West as Spring Begins",
        "date": "March 24, 2026",
        "tags": ["Snowpack", "Runoff", "Drought", "Water Supply"],
        "content": [
            {"text": "This year, we've seen record low snowpack. For the Upper Colorado River Basin, I wanted to take a look at how this year has compared to other low years in the record. Peak snowpack (most likely) occurred on March 9 at a meager 8.9\" of water in the snow. This is compared to a normal of 16\" around April 6 (median)."},
            {"text": "Surprisingly, this is not the earliest peak date that has occurred. That actually happened in 2015, when the peak was 11.6\" on March 8 before melting started. While that was also much below average, and very early, we don't talk much about 2015. Well, that also happened to be the year of the Miracle May that soundly busted the drought in the Colorado Headwaters and water supplies exceeded forecasts."},
            {"image": "assets/images/ucrb_swe_20260324.png", "caption": "Snow water equivalent (SWE) for the Upper Colorado River Basin for the 2025-2026 season compared to average and other low years. Data from NRCS."},
            {"text": "It's hard to look at this graphic and realize that this year is making 2002 and 2012 look like decent years, but here we are. Is there hope of a Miracle May? While we don't know for sure what the pattern will be for May, likely not. For Colorado and Utah, May 2015 is in the record books as the wettest May. With the Climate Prediction Center showing increased chances of above average temperatures and below average precipitation for the region in April, rapid melting will continue."},
            {"text": "The next record that will possibly be broken is melt out - when the snowpack reaches zero. Earliest dates of melt out stand at June 14 2002 and June 13 2012. Given the current level of snowpack, projections provided by NRCS (both high and low) indicate melt out likely before June 10. I'd also venture a guess that this could be the first time we see melt out in May."},
            {"text": "Regardless of what's to come, the situation is bleak. Water utilities are already calling for restrictions and extra conservation measures. In Aurora, CO, they are urging residents to hold off on turning on irrigation systems until May 1 and putting strict rules into place. This summer will be a true test to how we manage and respond to a possible worst-case scenario."},
            {"text": "Originally published on LinkedIn: https://www.linkedin.com/posts/climatebecky_this-year-weve-seen-record-low-snowpack-activity-7442604590052761601-ntJ9"}
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
    }
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
        elif "table" in block:
            # 1. Build the Header Row
            table_header = [
                html.Thead(html.Tr([html.Th(col) for col in block["table"]["headers"]]))
            ]
            
            # 2. Build the Data Rows
            table_rows = [
                html.Tbody([
                    html.Tr([html.Td(cell) for cell in row]) 
                    for row in block["table"]["rows"]
                ])
            ]
            
            # 3. Assemble and style the Bootstrap Table
            post_children.append(
                dbc.Table(
                    table_header + table_rows,
                    bordered=True,
                    striped=True,    # Alternating row colors
                    hover=True,      # Highlights row on mouse hover
                    responsive=True, # Adds a scrollbar on small screens if it's too wide
                    className="mb-4 shadow-sm bg-white"
                )
            )
            # --- TYPE 7: YOUTUBE VIDEO (NEW) ---
        elif "youtube" in block:
            post_children.append(
                html.Div(
                    html.Iframe(
                        # We extract just the video ID from the user and format the embed URL
                        src=f"https://www.youtube.com/embed/{block['youtube']}",
                        style={"width": "100%", "height": "450px", "border": "none", "borderRadius": "8px"},
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
                        allowFullScreen=True
                    ),
                    className="mb-4 shadow-sm"
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