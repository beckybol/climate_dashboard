import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, title='Media Interviews | Climate Becky')

# --- DATA: Media Appearances ---
# I've migrated the most recent items for you. 
# You can add the older ones by copying a line and filling in the details!
media_list = [
    {
        "title": "Colorado under statewide drought emergency",
        "outlet": "9News",
        "date": "June 4, 2026",
        "link": "https://www.9news.com/video/news/state/colorado-climate/colorado-under-statewide-drought-emergency/73-81a05f10-c158-4c7f-acda-5239cc882cfa",
        "type": "Video"
    },
    {
        "title": "Colorado faces significant snow deficit as ski season struggles",
        "outlet": "9News",
        "date": "Dec 28, 2025",
        "link": "https://www.9news.com/video/news/local/colorado-climate/colorado-faces-significant-snow-deficit-as-ski-season-struggles/73-861cadaf-1bb4-4f55-8db2-88e73f2fd6d2",
        "type": "Video"
    },
    {
        "title": "Colorado climate experts say hot and dry conditions this summer could push the state into drought status",
        "outlet": "KUNC Public Radio",
        "date": "Jul 2, 2024",
        "link": "https://www.kunc.org/news/2024-07-02/colorado-climate-experts-say-hot-and-dry-conditions-this-summer-could-push-the-state-into-drought-status",
        "type": "Radio"
    },
    {
        "title": "CSU climate report warns of more drought and wildfires",
        "outlet": "9News",
        "date": "Jan 8, 2024",
        "link": "https://www.9news.com/video/news/state/colorado-climate/73-861cadaf-1bb4-4f55-8db2-88e73f2fd6d2",
        "type": "Video"
    },
    {
        "title": "Colorado snowpack off to a below-average start",
        "outlet": "Denver Gazette",
        "date": "Dec 29, 2023",
        "link": "https://gazette.com/news/colorado-snowpack-off-to-a-below-average-start/article_bd63b5d8-a696-11ee-ab51-5393b55ecf00.html",
        "type": "Article"
    },
    {
        "title": "Another good winter of snow could help recharge the Colorado River",
        "outlet": "Aspen Daily News",
        "date": "Dec 10, 2023",
        "link": "https://www.aspendailynews.com/news/another-good-winter-of-snow-could-help-recharge-the-colorado-river/article_fb1df3b8-9703-11ee-b792-73ffddd1b23f.html",
        "type": "Article"
    },
    {
        "title": "Winter snow could help recharge the Colorado River. But what if it doesn't?",
        "outlet": "KUNC Public Radio",
        "date": "Nov 6, 2023",
        "link": "https://www.kunc.org/news/2023-11-06/winter-snow-could-help-recharge-the-colorado-river-but-what-if-it-doesnt",
        "type": "Radio"
    },
    {
        "title": "After a wet water year, can Colorado hope for a repeat? Not quite, experts say.",
        "outlet": "The Colorado Sun",
        "date": "Sep 27, 2023",
        "link": "https://coloradosun.com/2023/09/27/wet-water-year-colorado-repeat-not-quite/",
        "type": "Article"
    },
    {
        "title": "Report: West particularly hard hit by hotter fall temperatures",
        "outlet": "KUNR Public Radio",
        "date": "Sep 7, 2023",
        "link": "https://www.kunr.org/2023-09-07/hotter-fall-temperatures-west",
        "type": "Radio"
    },
    {
        "title": "Colorado natural disasters rise 275% over 20 years",
        "outlet": "Fox 31 Colorado",
        "date": "Aug 29, 2023",
        "link": "https://kdvr.com/news/local/colorado-natural-disasters-trending-up/",
        "type": "Video"
    },
    {
        "title": "What research is being done on wildfires at CSU?",
        "outlet": "CSU Source",
        "date": "Jul 31, 2023",
        "link": "https://source.colostate.edu/what-research-is-being-done-on-wildfires-at-csu/",
        "type": "Article"
    },
    {
        "title": "118-degree heat record in Bennett isn't valid",
        "outlet": "9News",
        "date": "Jul 11, 2023",
        "link": "https://www.9news.com/video/news/state/colorado-climate/118-degree-heat-record-in-bennett-isnt-valid/73-53f43894-ddf0-4f4c-8cb5-ae65ba080078",
        "type": "Video"
    },
    {
        "title": "Colorado is drought-free for the first time since 2019. Will it last?",
        "outlet": "Summit Daily",
        "date": "Jul 10, 2023",
        "link": "https://www.summitdaily.com/news/colorado-is-drought-free-for-the-first-time-since-2019-will-it-last/",
        "type": "Article"
    },
    {
        "title": "Extra snowfall this year means happy Colorado trees",
        "outlet": "KKCO News",
        "date": "Feb 4, 2023",
        "link": "https://www.nbc11news.com/2023/02/04/extra-snowfall-means-happy-trees/",
        "type": "Video"
    },
     {
        "title": "The Colorado River is overused and shrinking.",
        "outlet": "LA Times",
        "date": "Jan 26, 2023",
        "link": "https://www.latimes.com/environment/story/2023-01-26/colorado-river-in-crisis-the-west-faces-a-water-reckoning",
        "type": "Article"
    },
]

# --- HELPER FUNCTION ---
def make_media_row(item):
    """
    Creates a styled list item for a media appearance.
    Automatically assigns icons and colors based on the 'type'.
    """
    
    # Define style based on media type
    if item["type"] == "Video":
        icon = "bi-camera-reels-fill"
        color = "danger" # Red for video
    elif item["type"] == "Radio":
        icon = "bi-mic-fill"
        color = "info"   # Teal for audio
    else:
        icon = "bi-newspaper"
        color = "primary" # Blue for text
        
    return dbc.ListGroupItem(
        dbc.Row(
            [
                # Column 1: Icon & Date
                dbc.Col(
                    [
                        html.Div(
                            html.I(className=f"bi {icon} text-white"),
                            className=f"bg-{color} rounded-circle d-flex align-items-center justify-content-center shadow-sm",
                            style={"width": "40px", "height": "40px"}
                        ),
                    ],
                    width="auto", className="pe-3 d-flex align-items-center"
                ),
                
                # Column 2: Content
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Span(item["outlet"], className="fw-bold text-uppercase small text-muted spacing-1 me-2"),
                                html.Span("• " + item["date"], className="small text-muted")
                            ],
                            className="mb-1"
                        ),
                        html.H5(item["title"], className="mb-0 fw-bold text-dark"),
                    ],
                    className="d-flex flex-column justify-content-center"
                ),
                
                # Column 3: Arrow (Visual cue)
                dbc.Col(
                    html.I(className="bi bi-chevron-right text-muted"),
                    width="auto", className="d-flex align-items-center ms-auto"
                )
            ]
        ),
        action=True,
        href=item["link"],
        target="_blank",
        className="border-0 border-bottom py-4 px-3"
    )

# --- LAYOUT ---
layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                [
                    html.H1("In the Media", className="display-4 fw-bold text-primary mb-3"),
                    html.P(
                        "Recent interviews, news segments, and commentary on climate events.",
                        className="lead text-muted mb-5"
                    ),
                ],
                width=12, md=10
            )
        ),

        # The List
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.ListGroup(
                        [make_media_row(item) for item in media_list],
                        flush=True
                    ),
                    className="shadow border-0 mb-5"
                ),
                width=12, lg=10
            ),
            justify="center"
        )
    ],
    className="py-5"
)