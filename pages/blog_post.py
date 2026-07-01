import dash
from dash import callback, html, dcc
import dash_bootstrap_components as dbc
import urllib.parse
from data.blog_data import blog_posts, make_climate_post

# ==========================================
# 1. DYNAMIC META TAG FUNCTIONS
# ==========================================
# These functions allow Dash to build the correct social media preview 
# tags before the page even finishes loading!

# --- SIMPLIFIED META FUNCTIONS ---
def get_post_title(post_id=None, **kwargs):
    if not post_id: return "Climate Blog | Becky Bolinger"
    post = next((p for p in blog_posts if p["id"] == post_id), None)
    return f"{post['title']} | Becky Bolinger" if post else "Post Not Found"

def get_post_description(post_id=None, **kwargs):
    if not post_id: return "Climate updates and analysis."
    post = next((p for p in blog_posts if p["id"] == post_id), None)
    if post and "content" in post:
        # Safely find the first text block
        text_block = next((b for b in post["content"] if "text" in b), None)
        return text_block["text"][:150] + "..." if text_block else "Climate analysis."
    return "Climate updates and analysis."

def get_post_image(post_id=None, **kwargs):
    if not post_id: return "https://climatebecky.com/assets/images/co_swe_time.png"
    post = next((p for p in blog_posts if p["id"] == post_id), None)
    if post and "preview_image" in post:
        # Ensure absolute URL
        img = post["preview_image"]
        return img if img.startswith("http") else f"https://climatebecky.com/{img.lstrip('/')}"
    return "https://climatebecky.com/assets/images/co_swe_time.png"


# ==========================================
# 2. REGISTER PAGE (STAYS AT THE TOP!)
# ==========================================
dash.register_page(
    __name__, 
    path_template='/blog/<post_id>',
#    title="Blog | Climate Becky", # Use a static string here
    title=get_post_title(),  # Use the dynamic function for the title
    description=get_post_description(),  # Use the dynamic function for the description
#    description="Read the latest in-depth climate analysis and information."
)

# ==========================================
# 3. HELPER: SOCIAL SHARE BUTTONS
# ==========================================
def make_share_buttons(post_title, post_link):
    encoded_link = urllib.parse.quote(post_link, safe='')
    encoded_title = urllib.parse.quote(post_title, safe='')
    
    linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_link}"
    facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={encoded_link}"
    bluesky_url = f"https://bsky.app/intent/compose?text={encoded_title}%3A%20{encoded_link}"

    bluesky_icon = html.Img(src="/assets/bluesky.svg", style={"height": "18px", "width": "18px"})
    linkedin_icon = html.I(className="bi bi-linkedin", style={"fontSize": "18px", "color": "#0a66c2"})
    facebook_icon = html.I(className="bi bi-facebook", style={"fontSize": "18px", "color": "#1877f2"})

    btn_style = {
        "width": "32px", "height": "32px", "display": "flex", 
        "alignItems": "center", "justifyContent": "center",
        "border": "none", "background": "transparent"
    }

    return html.Div(
        [
            html.Small("Share: ", className="text-muted me-2 fw-bold"),
            dbc.Button(linkedin_icon, href=linkedin_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
            dbc.Button(bluesky_icon, href=bluesky_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
            dbc.Button(facebook_icon, href=facebook_url, target="_blank", style=btn_style, className="p-0 me-1 hover-scale"),
        ],
        className="d-flex align-items-center mt-5 pt-3 border-top"
    )

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
def layout(post_id=None, **kwargs):
    current_post = next((p for p in blog_posts if p["id"] == post_id), None)
    
    if not current_post:
        return dbc.Container(html.H1("Post not found!"), className="py-5")
    
    # Define URL stuff exactly once
    base_url = "https://climatebecky.com/blog"
    post_link = f"{base_url}/{post_id}"
    
    full_post_content = make_climate_post(current_post)
    share_buttons = make_share_buttons(current_post["title"], post_link)
    
    return dbc.Container(
        [
            dbc.Button([html.I(className="bi bi-arrow-left me-2"), "Back to Blog"], href="/blog", color="link", className="text-decoration-none mb-4 p-0"),
            dbc.Row(dbc.Col([full_post_content, share_buttons], width=12, lg=8, className="mx-auto"))
        ],
        fluid=True, className="py-5"
    )
