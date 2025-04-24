import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

# Load the data from the provided files
file_2023 = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/va/va_2023_precinct__ret_241218_utf.csv'
file_2024 = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/va/va_2024_precinct__ret_241218_utf.csv'

data_2023 = pd.read_csv(file_2023)
data_2024 = pd.read_csv(file_2024)

# Step 1: Filter 
## a) for rows that are either Democrat or Republican 
## b) and down to only the DistrictType (office) of interest
## c) to get rid of provisionals
valid_parties = ['Democratic', 'Republican']
district_type_2023 = 'state-senate'
office_title_2024 = 'President and Vice President'

filtered_2023 = data_2023[
    (data_2023['Party'].isin(valid_parties)) & 
    (data_2023['DistrictType'] == district_type_2023) &
    (~data_2023['PrecinctName'].str.lower().str.contains("provisional", case=False, na=False))
    ]
filtered_2024 = data_2024[
    (data_2024['Party'].isin(valid_parties)) & 
    (data_2024['OfficeTitle'] == office_title_2024) &
    (~data_2024['PrecinctName'].str.lower().str.contains("provisional", case=False, na=False))
    ]

# Step 2: Aggregate votes by precinct and party
aggregated_2023 = filtered_2023.groupby(['PrecinctName', 'Party'], as_index=False)['TOTAL_VOTES'].sum()
aggregated_2024 = filtered_2024.groupby(['PrecinctName', 'Party'], as_index=False)['TOTAL_VOTES'].sum()

# Step 3: Transform from long to wide format
wide_2023 = aggregated_2023.pivot(index='PrecinctName', columns='Party', values='TOTAL_VOTES').reset_index()
wide_2024 = aggregated_2024.pivot(index='PrecinctName', columns='Party', values='TOTAL_VOTES').reset_index()

# Ensure missing values are filled with 0
wide_2023 = wide_2023.fillna(0)
wide_2024 = wide_2024.fillna(0)

# Step 4: Merge the data files on precinct
merged_data = pd.merge(wide_2023, wide_2024, on='PrecinctName', how='outer', suffixes=('_2023', '_2024'))

# Step 5: Calculate two-way voteshare for Democrats (at Precinct Level)
merged_data['Democratic_Voteshare_2023'] = merged_data['Democratic_2023'] / (
    merged_data['Democratic_2023'] + merged_data['Republican_2023']
)
merged_data['Democratic_Voteshare_2024'] = merged_data['Democratic_2024'] / (
    merged_data['Democratic_2024'] + merged_data['Republican_2024']
)

# Replace any NaN values (e.g., where totals are 0) with 0
merged_data['Democratic_Voteshare_2023'] = merged_data['Democratic_Voteshare_2023'].fillna(0)
merged_data['Democratic_Voteshare_2024'] = merged_data['Democratic_Voteshare_2024'].fillna(0)

# Display results
# print(merged_data[['PrecinctName', 'Democratic_Voteshare_2023', 'Democratic_Voteshare_2024']].head())

# probably want to calculate voteshare by district 
def calculate_voteshare_by_district(merged_data, data_2023, district_type):
    """
    Calculates Democratic voteshare for 2023 and 2024 by DistrictType.

    Parameters:
        merged_data (pd.DataFrame): Dataframe containing merged precinct-level vote totals.
        data_2023 (pd.DataFrame): Dataframe containing the mapping of PrecinctName to DistrictName and DistrictType.
        district_type (str): The DistrictType to filter and aggregate (e.g., 'state-house').

    Returns:
        pd.DataFrame: Aggregated dataframe with Democratic voteshare columns.
    """
    # Step 1: Filter data_2023 for the specified DistrictType
    district_mapping = data_2023[data_2023['DistrictType'] == district_type][['PrecinctName', 'DistrictName']].drop_duplicates()

    # Step 2: Merge district mapping into merged_data
    merged_with_district = pd.merge(
        merged_data,
        district_mapping,
        on='PrecinctName',
        how='left'
    )

    # Step 3: Aggregate votes at the DistrictName level
    aggregated = merged_with_district.groupby('DistrictName', as_index=False).agg({
        'Democratic_2023': 'sum',
        'Republican_2023': 'sum',
        'Democratic_2024': 'sum',
        'Republican_2024': 'sum'
    })

    # Step 4: Calculate two-way voteshare for Democrats
    aggregated['Democratic_Voteshare_2023'] = aggregated['Democratic_2023'] / (
        aggregated['Democratic_2023'] + aggregated['Republican_2023']
    )
    aggregated['Democratic_Voteshare_2024'] = aggregated['Democratic_2024'] / (
        aggregated['Democratic_2024'] + aggregated['Republican_2024']
    )

    # Replace NaN values with 0 (e.g., if no votes are recorded in a district)
    aggregated['Democratic_Voteshare_2023'] = aggregated['Democratic_Voteshare_2023'].fillna(0)
    aggregated['Democratic_Voteshare_2024'] = aggregated['Democratic_Voteshare_2024'].fillna(0)
    aggregated['Change_in_2024'] = aggregated['Democratic_Voteshare_2024'] - aggregated['Democratic_Voteshare_2023']

    # Return the result with only relevant columns
    return aggregated[['DistrictName', 'Democratic_Voteshare_2023', 'Democratic_Voteshare_2024','Change_in_2024']]

by_state_senate = calculate_voteshare_by_district(merged_data, data_2023, 'state-senate')

def remove_uncontested_districts(df,col="",threshold=0.05):
    return df[(df[col] >= threshold) & (df[col] <= 1 - threshold)]

by_state_senate = remove_uncontested_districts(by_state_senate,'Democratic_Voteshare_2023')

def plot_voteshare_comparison(df,district_col='DistrictName'):
    """
    Plots a scatter plot comparing Democratic vote share in 2023 (y-axis) vs. 2024 (x-axis).

    Parameters:
        df (pd.DataFrame): Dataframe containing 'Democratic_Voteshare_2023' and 'Democratic_Voteshare_2024' columns.
    """
    # Check if required columns are in the dataframe
    if 'Democratic_Voteshare_2023' not in df.columns or 'Democratic_Voteshare_2024' not in df.columns:
        raise ValueError("Dataframe must contain 'Democratic_Voteshare_2023' and 'Democratic_Voteshare_2024' columns.")

    # Calculate the mean vote shares for 2023 and 2024
    mean_2023 = df['Democratic_Voteshare_2023'].mean()
    mean_2024 = df['Democratic_Voteshare_2024'].mean()

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Democratic_Voteshare_2024'], df['Democratic_Voteshare_2023'], alpha=0.6)
    plt.title("Democratic Vote Share: 2023 vs. 2024", fontsize=14)
    plt.xlabel("Democratic Vote Share 2024", fontsize=12)
    plt.ylabel("Democratic Vote Share 2023", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Vote Share (2023)')
    plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='50% Vote Share (2024)')
    plt.axvline(x=mean_2023, color='purple', linestyle='--', alpha=0.8, label=f'Average 2023 Perf ({mean_2023:.2f})')
    plt.axvline(x=mean_2024, color='green', linestyle='--', alpha=0.8, label=f'Average 2024 Perf ({mean_2024:.2f})')
    
    plt.legend()
    # Add labels for districts
    for _, row in df.iterrows():
        plt.text(
            row['Democratic_Voteshare_2024'],
            row['Democratic_Voteshare_2023'],
            str(row[district_col]),
            fontsize=8,
            alpha=0.7
        )
    plt.show()
    """
    Plots a scatter plot comparing Democratic vote share in 2024 (x-axis) vs. the change in vote share from 2023 to 2024 (y-axis),
    with a superimposed line to indicate a tie in 2023 (vote share is 50%).

    Parameters:
        df (pd.DataFrame): Dataframe containing 'Democratic_Voteshare_2024' and 'Change_in_2024' columns.
    """
    # Check if required columns are in the dataframe
    if 'Democratic_Voteshare_2024' not in df.columns or 'Change_in_2024' not in df.columns:
        raise ValueError("Dataframe must contain 'Democratic_Voteshare_2024' and 'Change_in_2024' columns.")

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Democratic_Voteshare_2024'], df['Change_in_2024'], alpha=0.6, label='Districts')
    plt.title("Change in Democratic Vote Share (2023 to 2024) vs. 2024 Vote Share", fontsize=14)
    plt.xlabel("Democratic Vote Share 2024", fontsize=12)
    plt.ylabel("Change in Vote Share (2023 to 2024)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add reference lines
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='50% Vote Share (2024)')
    
    # Add tie line for 2023 (Change in Vote Share = 50% - Democratic_Voteshare_2024)
    x_vals = df['Democratic_Voteshare_2024']
    tie_line_y = 0.5 - x_vals
    plt.plot(x_vals, tie_line_y, color='green', linestyle='--', alpha=0.8, label='2023 Tie Line (50%)')
    
    # Add legend and display
    plt.legend()
    plt.show()


def plot_voteshare_with_polygon(df,district_col='DistrictName'):
    """
    Plots a scatter plot comparing Democratic vote share in 2024 (x-axis) vs. the change in vote share from 2023 to 2024 (y-axis),
    with updated tie line, average 2023 and 2024 performance lines, and opportunity region as a fully bound polygon.

    Parameters:
        df (pd.DataFrame): Dataframe containing 'Democratic_Voteshare_2024', 'Change_in_2024', and 'Democratic_Voteshare_2023' columns.
    """
    # Check if required columns are in the dataframe
    required_columns = ['Democratic_Voteshare_2024', 'Change_in_2024', 'Democratic_Voteshare_2023', district_col]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Dataframe must contain '{column}' column.")

    # Calculate the mean vote shares for 2023 and 2024
    mean_2023 = df['Democratic_Voteshare_2023'].mean()
    mean_2024 = df['Democratic_Voteshare_2024'].mean()

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Color points based on 2023 Democratic vote share
    above_50 = df[df['Democratic_Voteshare_2023'] > 0.5]
    below_50 = df[df['Democratic_Voteshare_2023'] <= 0.5]

    plt.scatter(
        above_50['Democratic_Voteshare_2024'],
        above_50['Change_in_2024'],
        color='lightblue',
        alpha=0.6,
        label='2023 Vote Share > 50%'
    )
    plt.scatter(
        below_50['Democratic_Voteshare_2024'],
        below_50['Change_in_2024'],
        color='pink',
        alpha=0.6,
        label='2023 Vote Share ≤ 50%'
    )

    # Add title and labels
    plt.title("Change in Democratic Vote Share (2023 to 2024) vs. 2024 Vote Share", fontsize=14)
    plt.xlabel("Democratic Vote Share 2024", fontsize=12)
    plt.ylabel("Change in Vote Share (2023 to 2024)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add labels for districts
    for _, row in df.iterrows():
        plt.text(
            row['Democratic_Voteshare_2024'],
            row['Change_in_2024'],
            str(row[district_col]),
            fontsize=8,
            alpha=0.7
        )

    # Add reference lines
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Change')
    plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='50% Vote Share (2024)')
    plt.axvline(x=mean_2023, color='purple', linestyle='--', alpha=0.8, label=f'Average 2023 Perf ({mean_2023:.2f})')
    plt.axvline(x=mean_2024, color='green', linestyle='--', alpha=0.8, label=f'Average 2024 Perf ({mean_2024:.2f})')

    # Add tie line for 2023 (Change in Vote Share = Democratic_Voteshare_2024 - 0.5)
    x_vals = df['Democratic_Voteshare_2024']
    tie_line_y = x_vals - 0.5
    plt.plot(
        x_vals, tie_line_y, 
        color='lightgrey', linestyle=':', alpha=0.8, linewidth=1, 
        label='2023 Tie Line (50%)'
    )

    ceiling = 0.1 # can automate placement based on data later

    vertices = [(0.5,0),[0.5, ceiling],[ceiling+0.5, ceiling]]
    polygon = Polygon(vertices, closed=True, color='orange', alpha=0.3, label='Opportunity Region')
    plt.gca().add_patch(polygon)

    # Add legend and display
    plt.legend()
    plt.show()

plot_voteshare_comparison(by_state_senate)
