# In pages/home.py
import dash
from dash import html

# The path='/' makes this the actual Homepage
dash.register_page(__name__, path='/') 

layout = html.Div([
    html.H1("Welcome to My Portfolio"),
    html.P("Select a link above to explore my work."),
    # You could put a nice hero image or intro text here
])