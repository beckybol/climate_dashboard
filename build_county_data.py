import pandas as pd
import numpy as np
import os

print("Processing nClimDiv County Data...")
current_dir = os.path.dirname(os.path.abspath(__file__))

# UPDATE THESE FILENAMES to match the ones you downloaded from NCEI
tmp_file = os.path.join(current_dir, 'data', 'climdiv-tmpccy-v1.0.0-20260506.txt')
pcp_file = os.path.join(current_dir, 'data', 'climdiv-pcpncy-v1.0.0-20260506.txt')

# NCDC uses 1-48 for contiguous US states. We must map these to standard 2-digit FIPS codes for Plotly.
ncdc_to_fips = {
    '01': '01', '02': '04', '03': '05', '04': '06', '05': '08', '06': '09', '07': '10', '08': '12',
    '09': '13', '10': '16', '11': '17', '12': '18', '13': '19', '14': '20', '15': '21', '16': '22',
    '17': '23', '18': '24', '19': '25', '20': '26', '21': '27', '22': '28', '23': '29', '24': '30',
    '25': '31', '26': '32', '27': '33', '28': '34', '29': '35', '30': '36', '31': '37', '32': '38',
    '33': '39', '34': '40', '35': '41', '36': '42', '37': '44', '38': '45', '39': '46', '40': '47',
    '41': '48', '42': '49', '43': '50', '44': '51', '45': '53', '46': '54', '47': '55', '48': '56'
}

def parse_climdiv(filepath, value_name):
    """Parses the fixed-width climdiv file and extracts March values."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            state_code = line[0:2]
            if state_code not in ncdc_to_fips:
                continue # Skip non-CONUS
            
            county_code = line[2:5]
            fips = ncdc_to_fips[state_code] + county_code
            year = int(line[7:11])
            
            # March is columns 26-32 (index 25 to 32)
            mar_val = float(line[25:32].strip())
            
            # Missing data in nClimDiv is often -99.99 or -9.99
            if mar_val < -90:
                mar_val = np.nan
                
            records.append({'FIPS': fips, 'Year': year, value_name: mar_val})
    return pd.DataFrame(records)

# Parse both files
df_tmp = parse_climdiv(tmp_file, 'Temp')
df_pcp = parse_climdiv(pcp_file, 'Precip')

# Merge them
df = pd.merge(df_tmp, df_pcp, on=['FIPS', 'Year'])

# Calculate 1991-2020 Normals and Anomalies
normals = df[(df['Year'] >= 1991) & (df['Year'] <= 2020)].groupby('FIPS').mean().reset_index()
normals = normals.rename(columns={'Temp': 'Temp_Normal', 'Precip': 'Precip_Normal'})

df = pd.merge(df, normals[['FIPS', 'Temp_Normal', 'Precip_Normal']], on='FIPS')
df['Temp_Anomaly'] = df['Temp'] - df['Temp_Normal']

# Find Counties where 2026 was the warmest on record
max_years = df.loc[df.groupby('FIPS')['Temp'].idxmax()]
warmest_2026_fips = max_years[max_years['Year'] == 2026]['FIPS'].tolist()

# Save datasets
out_history = os.path.join(current_dir, 'data', 'county_march_history.csv')
df.to_csv(out_history, index=False)

print(f"Processed {len(df['FIPS'].unique())} counties.")
print(f"Found {len(warmest_2026_fips)} counties where March 2026 was the warmest on record.")
print("Data saved. Ready for Dash!")