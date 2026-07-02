# pages/blog.py
from datetime import datetime

import dash
from dash import html
import dash_bootstrap_components as dbc
from data.blog_data import blog_posts
from datetime import datetime, timedelta

dash.register_page(__name__, path='/blog', title='Blog | Climate Becky', description='Becky\'s Blog: Climate Change, Weather, and Climate Variability Updates')

def create_snippet_card(post):
    # Try to grab the first text block for the snippet
    snippet = "Click to read more about this update..."
    for block in post["content"]:
        if "text" in block:
            snippet = block["text"][:180] + "..."
            break

    # 1. Parse the post date (Assuming format "Month Day, Year")
    post_date = datetime.strptime(post["date"], "%B %d, %Y")
    
    # 2. Check if the post is less than 14 days old
    is_new = datetime.now() - post_date < timedelta(days=14)
    
    # 3. Create the "New" badge if applicable
    new_badge = dbc.Badge("NEW!", color="warning", className="ms-2") if is_new else None

    # Update your title line to include the badge
    title_row = html.Div([
        html.H4(post["title"], className="card-title text-primary fw-bold d-inline"),
        new_badge
    ],
    className="d-flex align-items-center mb-2")  # Aligns title and badge nicely
    
    # Build the main text body of the preview
    card_body_content = [
        title_row,
        html.H6(post["date"], className="card-subtitle mb-2 text-muted"),
        html.P(snippet, className="card-text mt-2", style={"fontSize": "0.95rem"}),
        dbc.Button("Read More", href=f"/blog/{post['id']}", color="outline-primary", size="sm")
    ]

    # Check if a preview image is defined for this post
    if "preview_image" in post:
        # Use a horizontal layout: Image on the left (or right), text on the other side
        row_content = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        style={
                            "backgroundImage": f"url('{post['preview_image']}')",
                            "backgroundSize": "cover",
                            "backgroundPosition": "center",
                            "height": "100%",
                            "minHeight": "150px",
                            "borderRadius": "4px 0 0 4px" if not dash.get_asset_url else "4px" # rounded corners
                        },
                        className="h-100"
                    ),
                    width=12, md=4, className="pe-md-0"
                ),
                dbc.Col(dbc.CardBody(card_body_content), width=12, md=8)
            ],
            className="g-0 d-flex align-items-stretch" # g-0 removes gaps between image and text columns
        )
    else:
        # Fallback to standard full-width card body if no image is defined
        row_content = dbc.CardBody(card_body_content)

    return dbc.Card(row_content, className="mb-4 shadow-sm border-0 overflow-hidden")

layout = dbc.Container(
    [
        html.H1("Climate Blog", className="display-4 fw-bold mb-5 text-center"),
        dbc.Row(
            dbc.Col(
                [create_snippet_card(post) for post in blog_posts],
                width=12, lg=9, xl=8, className="mx-auto" 
            )
        )
    ],
    fluid=True,
    className="py-5"
)