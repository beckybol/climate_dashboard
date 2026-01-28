import pandas as pd
import os

# 1. Define your file paths
# Replace these with your actual filenames
files = {
    'Average Temp': 'cag_tavg_data.csv',
    'Max Temp': 'cag_tmax_data.csv',
    'Min Temp': 'cag_tmin_data.csv',
    'Precipitation': 'cag_pcp_data.csv'
}

data_frames = []

# 2. Loop through files and standardize them
for variable_name, file_path in files.items():
    # Load the CSV
    # Assumption: Your CSVs have columns like: 'Date', 'Alabama', 'Alaska', 'Arizona'...
    # If the first column is Date, we set index_col=0
    df = pd.read_csv(file_path)
    
    # 3. "Melt" the data from Wide to Long
    # This turns 50 state columns into one "State" column
    # id_vars should be your Date column. Change 'Date' if your column is named 'YearMonth' or similar.
    df_long = df.melt(id_vars=['time'], var_name='State', value_name='Value')
    
    # 4. Add a column identifying which variable this is (e.g., "Max Temp")
    df_long['Variable'] = variable_name
    
    # Append to our list
    data_frames.append(df_long)

# 5. Concatenate all 4 variables into one Master DataFrame
master_df = pd.concat(data_frames, ignore_index=True)

# 6. Data Cleanup & Helper Columns
# Convert Date column to actual datetime objects (critical for sorting/plotting)
master_df['Date'] = pd.to_datetime(master_df['time'], format='%Y-%m-%d') # Adjust format if needed

# Create the "Month" helper column for your filtering requirement
master_df['Month'] = master_df['Date'].dt.month
master_df['Year'] = master_df['Date'].dt.year

# 7. (Optional) State ID Mapping
# Plotly maps often work best with 2-letter codes (AL, AK, etc.) rather than full names.
# You might need a dictionary to map full names to abbreviations if your CSVs use full names.
state_map = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Apply map only if your CSV has full names. If it already has codes, skip this.
master_df['State_Code'] = master_df['State'].map(state_map)

# 8. Check the output
print(master_df.head())
print(master_df.info())

# Save to use in your app
master_df.to_csv('master_climate_data.csv', index=False)