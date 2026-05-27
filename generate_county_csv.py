import json
import pandas as pd
import urllib.request

# The URL from your code
url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'

print("Downloading GeoJSON...")
with urllib.request.urlopen(url) as response:
    counties_data = json.load(response)

print("Extracting county names...")
county_list = []

# Iterate through the GeoJSON features
for feature in counties_data['features']:
    props = feature.get('properties', {})
    
    # Note: 'NAME' and 'FIPS' are the standard keys in this dataset
    # We use .get() to avoid errors if a key is missing
    fips = props.get('FIPS')
    name = props.get('NAME')
    
    if fips and name:
        county_list.append({
            'FIPS': str(fips).zfill(5), 
            'CountyName': name
        })

# Create DataFrame and save
df_counties = pd.DataFrame(county_list)
df_counties.to_csv('county_names.csv', index=False)

print("Successfully saved county_names.csv!")
print(df_counties.head())