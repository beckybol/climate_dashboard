import pandas as pd
import requests
import time
import os

print("Loading station inventory...")
current_dir = os.path.dirname(os.path.abspath(__file__))
inv_file = os.path.join(current_dir, 'data', 'mly_inventory.txt')

# 1. Load Inventory (Fixed-Width Format)
col_specs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)]
names = ['id', 'latitude', 'longitude', 'elevation', 'state', 'name']
inv_df = pd.read_fwf(inv_file, colspecs=col_specs, header=None, names=names)

# 2. Filter for Contiguous US
conus_states = [
    'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 
    'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 
    'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]
conus_df = inv_df[inv_df['state'].isin(conus_states)]

print(f"Found {len(conus_df)} CONUS stations. Beginning API queries (this will take a few minutes)...")

record_stations = []
count = 0

# 3. Loop through stations and check ACIS
for index, row in conus_df.iterrows():
    sid = row['id']
    count += 1
    
    if count % 500 == 0:
        print(f"Processed {count}/{len(conus_df)} stations...")
    
    # Request monthly max temperatures for the entire period of record
    payload = {
        "sid": sid,
        "sdate": "por",
        "edate": "2026-03-31",
        "elems": [{"name": "maxt", "interval": "mly", "duration": "mly", "reduce": "max"}]
    }
    
    try:
        r = requests.post("http://data.rcc-acis.org/StnData", json=payload, timeout=10)
        data = r.json()
        
        if "data" not in data:
            continue
            
        march_temps = []
        march_2026_temp = None
        
        # Filter down to only March records
        for item in data['data']:
            date_str = item[0]
            val_str = item[1]
            
            if date_str.endswith("-03"): # Look only at March
                try:
                    val = float(val_str)
                    march_temps.append(val)
                    if date_str == "2026-03":
                        march_2026_temp = val
                except ValueError:
                    pass # Skip missing 'M' data
        
        if not march_temps or march_2026_temp is None:
            continue
            
        # Find the all-time high for March
        all_time_march_max = max(march_temps)
        
        # Did March 2026 break or tie the record? (And require at least 30 years of data for it to be a robust "record")
        if march_2026_temp >= all_time_march_max and len(march_temps) >= 30:
            record_stations.append({
                "GHCN_ID": sid,
                "name": row['name'].strip(),
                "state": row['state'],
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "elevation": row['elevation'],
                "record_temp": march_2026_temp,
                "years_on_record": len(march_temps)
            })
            
    except Exception as e:
        pass # Silently skip timeout errors
        
    time.sleep(0.05) # Be nice to the API

# 4. Save the results
out_file = os.path.join(current_dir, 'data', 'march_2026_heat_records.csv')
results_df = pd.DataFrame(record_stations)
results_df.to_csv(out_file, index=False)
print(f"Done! Saved {len(results_df)} record-breaking stations to data/march_2026_heat_records.csv")