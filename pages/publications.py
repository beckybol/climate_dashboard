import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, title='Publications | Climate Becky')

# --- DATA: Scientific Publications & Reports ---
# Easy to update! Just copy a block and fill in the new info.
scientific_pubs = [
    {
        "title": "3rd Edition of the Climate Change in Colorado Report",
        "venue": "Colorado State University",
        "year": "2024",
        "link": "https://climatechange.colostate.edu",
        "type": "Report",
        "authors": "Becky Bolinger, et al.",
        "desc": "A comprehensive analysis of recent trends in Colorado's climate and hydrology, with a focus on extreme events like heat waves, droughts, and wildfires."
    },
    {
        "title": "An assessment of the extremes and impacts of the February 2021 South-Central U.S. Arctic outbreak",
        "venue": "Weather and Climate Extremes",
        "year": "2022",
        "link": "https://doi.org/10.1016/j.wace.2022.100461",
        "type": "Journal Article",
        "authors": "R.A. Bolinger and Coauthors"
    },
    {
        "title": "A CONUS-wide standardized precipitation-evapotranspiration index for major U.S. row crops",
        "venue": "Journal of Hydrometeorology",
        "year": "2021",
        "link": "https://doi.org/10.1175/JHM-D-20-0270.1",
        "type": "Journal Article",
        "authors": "P.E. Goble, R.A. Bolinger, and R. S. Schumacher"
    },
    {
        "title": "Application of the NMME in the Development of a New Regional Seasonal Climate Forecast Tool",
        "venue": "Bulletin of the American Meteorological Society",
        "year": "2017",
        "link": "#",
        "type": "Journal Article",
        "authors": "R.A. Bolinger, A. D. Gronewold, K. Kompoltowicz, and L. M. Fry"
    },
    {
        "title": "Attribution and Characteristics of Wet and Dry Seasons in the Upper Colorado River Basin",
        "venue": "Journal of Climate",
        "year": "2014",
        "link": "#",
        "type": "Journal Article",
        "authors": "R.A. Bolinger, C. D. Kummerow, and N. J. Doesken"
    },
     {
        "title": "A Comparison of in Situ, Reanalysis, and Satellite Water Budgets over the Upper Colorado River Basin",
        "venue": "Journal of Hydrometeorology",
        "year": "2013",
        "link": "#",
        "type": "Journal Article",
        "authors": "R.A. Smith (Bolinger) and C. D. Kummerow"
    }
]

# --- DATA: Media & Articles ---
media_articles = [
    {"title": "Great Plains could see its most significant drought in a decade", "source": "Washington Post", "date": "Apr 2022", "link": "https://www.washingtonpost.com/weather/2022/04/11/drought-plains-fire-us/"},
    {"title": "Celebrate Women's History Month with six inspiring women in atmospheric sciences", "source": "Washington Post", "date": "Mar 2022", "link": "https://www.washingtonpost.com/weather/2022/03/08/women-science-celebrate-atmospheric-meteorologist/"},
    {"title": "U.S. has experienced fewer cold winter days than normal so far, a sign of climate warming", "source": "Washington Post", "date": "Feb 2022", "link": "https://www.washingtonpost.com/weather/2022/02/23/winter-temperatures-warming-2022/"},
    {"title": "Drought conditions improve in U.S. West, but more snow is needed", "source": "Washington Post", "date": "Jan 2022", "link": "https://www.washingtonpost.com/weather/2022/01/22/drought-west-us-improves/"},
    {"title": "La Niña, climate change, and bad luck: the climate context of Colorado's Marshall Fire", "source": "NOAA Climate.gov", "date": "Jan 2022", "link": "https://www.climate.gov/news-features/event-tracker/la-niña-climate-change-and-bad-luck-climate-context-colorado’s-marshall"},
    {"title": "How extreme climate conditions fueled unprecedented Colorado fire", "source": "Washington Post", "date": "Dec 2021", "link": "https://www.adn.com/nation-world/2021/12/31/how-extreme-climate-conditions-fueled-unprecedented-colorado-fire/"},
    {"title": "Snowpack is off to a poor start in the West, bad news amid widespread drought", "source": "Washington Post", "date": "Dec 2021", "link": "https://www.washingtonpost.com/weather/2021/12/07/snow-west-us-low-drought/"},
    {"title": "Depleted by drought, Lakes Powell and Mead were doomed from the beginning", "source": "Washington Post", "date": "Sep 2021", "link": "https://www.washingtonpost.com/weather/2021/09/10/lake-powell-mead-drought-history/"},
    {"title": "Warming is clearly visible in new US 'climate normal' datasets", "source": "The Conversation", "date": "May 2021", "link": "https://theconversation.com/warming-is-clearly-visible-in-new-us-climate-normal-datasets-159684"},
    {"title": "Did the March 2021 snowstorm improve drought conditions across Colorado?", "source": "Weather5280", "date": "Mar 2021", "link": "https://www.weather5280.com/2021/03/18/did-the-march-2021-snowstorm-improve-drought-conditions-across-colorado"},
]


# --- HELPER FUNCTIONS ---

def make_scientific_card(pub):
    """Creates a card for a scientific paper/report."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        dbc.Badge(pub["type"], color="primary" if pub["type"] == "Report" else "info", className="me-2"),
                        html.Small(pub["year"], className="text-muted")
                    ],
                    className="mb-2"
                ),
                html.H5(
                    html.A(pub["title"], href=pub["link"], target="_blank", className="text-decoration-none text-dark stretched-link"),
                    className="card-title fw-bold"
                ),
                html.H6(pub["venue"], className="card-subtitle mb-2 text-primary"),
                html.P(pub["authors"], className="card-text text-muted small fst-italic"),
                html.P(pub.get("desc", ""), className="card-text small") if "desc" in pub else None,
            ]
        ),
        className="mb-3 shadow-sm border-0 h-100 hover-shadow"
    )

def make_media_item(item):
    """Creates a list item for a media article."""
    # Determine badge color based on source
    color_map = {
        "Washington Post": "dark",
        "Weather5280": "warning",
        "NOAA Climate.gov": "primary",
        "The Conversation": "danger"
    }
    badge_color = color_map.get(item["source"], "secondary")
    
    return dbc.ListGroupItem(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Badge(item["source"], color=badge_color, className="me-2"),
                                html.Small(item["date"], className="text-muted")
                            ],
                            className="mb-1"
                        ),
                        html.H6(item["title"], className="mb-0 fw-bold"),
                    ],
                    width=10
                ),
                dbc.Col(
                    html.A(
                        html.I(className="bi bi-box-arrow-up-right text-muted"), 
                        href=item["link"], 
                        target="_blank",
                        className="stretched-link"
                    ),
                    width=2, className="text-end d-flex align-items-center justify-content-end"
                )
            ]
        ),
        action=True,  # Makes the whole row clickable and adds hover effect
        className="border-0 border-bottom py-3"
    )


# --- MAIN LAYOUT ---
layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("Publications & Media", className="display-4 fw-bold text-primary mb-4"),
                    html.P(
                        "A selection of my peer-reviewed research, official reports, and science communication articles.",
                        className="lead text-muted"
                    ),
                    html.Hr(className="my-5")
                ],
                width=12
            )
        ),

        # Section 1: Scientific Works
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3([html.I(className="bi bi-journal-text me-2"), "Scientific Papers & Reports"], className="mb-4"),
                        dbc.Row(
                            [dbc.Col(make_scientific_card(pub), md=6, lg=6) for pub in scientific_pubs],
                            className="g-4" # Grid gap
                        )
                    ],
                    width=12, className="mb-5"
                )
            ]
        ),

        # Section 2: In the Media
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3([html.I(className="bi bi-newspaper me-2"), "Selected Articles & Media"], className="mb-4"),
                        dbc.Card(
                            dbc.ListGroup(
                                [make_media_item(item) for item in media_articles],
                                flush=True # Removes outer borders for a cleaner look
                            ),
                            className="shadow-sm border-0"
                        )
                    ],
                    width=12, lg=10
                )
            ],
            justify="center" # Centers the list on the page
        )
    ],
    fluid=False,
    className="py-5"
)