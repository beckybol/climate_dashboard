# data/chart_builders.py
import pandas as pd
import plotly.express as px
import os

def build_snow_records_map():
    """Builds the Feb 2026 snow records map."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    records_csv = os.path.join(current_dir, 'snow_records_202602.csv')
    
    try:
        df_records = pd.read_csv(records_csv)
        fig = px.scatter_mapbox(
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
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except FileNotFoundError:
        return px.scatter(title="Data File Not Found")

# --- THE REGISTRY ---
# A dictionary that links a simple string name to the function itself
CHART_REGISTRY = {
    "feb_2026_snow_map": build_snow_records_map,
    # As you write more posts, you just add new functions here:
    # "spring_2026_outlook_table": build_spring_table,
    # "drought_monitor_animation": build_drought_animation,
}