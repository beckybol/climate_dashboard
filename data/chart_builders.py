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

# data/chart_builders.py
# ... existing imports ...

def build_wildfire_map():
    """Builds the interactive wildfire map."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure wildfires.csv is in the same directory as chart_builders.py
    csv_path = os.path.join(current_dir, 'wildfires.csv')
    
    try:
        df = pd.read_csv(csv_path)
        df['Visual Size'] = df['Current Size'] + 5000
        df['Size'] = df['Current Size'].apply(lambda x: f"{x:,} acres")
        
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            size="Visual Size",
            color="Current Size", # Color helps visually distinguish sizes
            hover_name="Fire",
            hover_data={
                "Latitude": False, 
                "Longitude": False,
                "Visual Size": False,
                "Start Date": True,
                "Current Size": False,
                "Size": True,
                "Containment": True,
                "Note": True
            },
            color_continuous_scale=px.colors.sequential.Sunsetdark_r,
            size_max=20,
            zoom=5,
            center={"lat": 38.5, "lon": -111.0}, # Centered between UT and CO
            mapbox_style="carto-positron",
            title="Active Wildfires (June 2026)"
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        return px.scatter(title=f"Error loading map: {e}")

# --- THE REGISTRY ---
# A dictionary that links a simple string name to the function itself
CHART_REGISTRY = {
    "feb_2026_snow_map": build_snow_records_map,
    "jun_2026_wildfire_map": build_wildfire_map,
    # As you write more posts, you just add new functions here:
    # "spring_2026_outlook_table": build_spring_table,
    # "drought_monitor_animation": build_drought_animation,
}