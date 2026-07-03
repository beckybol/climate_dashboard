import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/about', title='About Me | Climate Becky')

layout = dbc.Container([
    dbc.Row([
        # Column 1: Image
        dbc.Col([
            html.Img(src="assets/04021_00571.jpg", className="img-fluid shadow p-2 border"),
            html.Img(src="assets/logo.png", height="150px", className="mt-3"),
            dbc.Stack(
            [
                dbc.Button([html.I(className="bi bi-filetype-pdf me-2"), "download CV"], href=dash.get_asset_url("cv_bolinger_202604.pdf"), target="_blank", color="outline-dark", size="md", className="me-1"),
                dbc.Button([html.I(className="bi bi-pencil-square me-2"), "publications"], href="/publications", color="outline-dark", size="md", className="me-1"),
                dbc.Button([html.I(className="bi bi-chat-left-dots me-2"), "media interviews"], href="/media", color="outline-dark", size="md", className="me-1"),
            ],
            direction="vertical",
        )

        ], width=12, md=4, className="text-center mb-4"),
        # Column 2: Bio
        dbc.Col([
            html.P(
                """
                I'm a climatologist with 21 years of experience. I got my Bachelor's degree in Meteorology from Metro State University of Denver and my Master's degree in Meteorology from Florida State University. At FSU, I studied under the Florida State Climatologist and researched temperature trends across the southeast U.S. After graduating, I worked as a research climatologist at the University of Illinois. I returned to Colorado to complete my PhD at Colorado State University. After a 2-year postdoctoral fellowship with the NOAA Great Lakes Environmental Research Laboratory in Michigan, I spent 8 years at the Colorado Climate Center as the Assistant State Climatologist. I transitioned to consulting and spent almost 2 years as a Senior Climate Scientist at Lynker. I now work as an independent consultant and am passionate about communicating current climate issues and building interactive tools that make complex climate science accessible, understandable, and usable.
                """,
                className="lead"
            ),
            html.Hr(),
            html.H5("Skills"),
            html.Div([
                dbc.Badge("Climate Analysis", color="primary", className="me-2 p-2"),
                dbc.Badge("Drought Monitoring", color="primary", className="me-2 p-2"),
                dbc.Badge("Climate Change Science", color="primary", className="me-2 p-2"),
                dbc.Badge("Project Management", color="primary", className="me-2 p-2"),
                dbc.Badge("Stakeholder Engagement", color="primary", className="me-2 p-2"),
                dbc.Badge("Science Communication", color="primary", className="me-2 p-2"),
                dbc.Badge("Python", color="info", className="me-2 p-2"),
                dbc.Badge("Git / GitHub", color="info", className="me-2 p-2"),
                dbc.Badge("R", color="info", className="me-2 p-2"),
                dbc.Badge("HTML / CSS", color="info", className="me-2 p-2"),
                dbc.Badge("JavaScript", color="info", className="me-2 p-2"),
                dbc.Badge("PRISM", color="success", className="me-2 p-2"),
                dbc.Badge("nClimGrid", color="success", className="me-2 p-2"),
                dbc.Badge("ACIS Web Services", color="success", className="me-2 p-2"),
                dbc.Badge("Reanalysis Datasets", color="success", className="me-2 p-2"),
                dbc.Badge("Probabilistic Statistical Modeling", color="danger", className="me-2 p-2"),
                dbc.Badge("Model Validation", color="danger", className="me-2 p-2"),
                dbc.Badge("Statistical Downscaling", color="danger", className="me-2 p-2"),
                dbc.Badge("Plotly", color="warning", className="me-2 p-2"),
                dbc.Badge("High Charts", color="warning", className="me-2 p-2"),
                dbc.Badge("ggplot2", color="warning", className="me-2 p-2"),
                dbc.Badge("QGIS", color="warning", className="me-2 p-2"),
            ], className="d-flex flex-wrap gap-2")  # Wrap badges to new lines on smaller screens
        ], width=12, md=8)
    ], className="align-items-center mt-5")
])