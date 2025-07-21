# import pandas as pd
# import glob
# import os

# input_folder = 'data/'
# output_folder = 'classified_results/'

# os.makedirs(output_folder, exist_ok=True)

# def classify_rainfall(val):
#     if val == 0:
#         return 1
#     elif 0 < val < 10:
#         return 2
#     elif 10 <= val < 50:
#         return 3
#     elif 50 <= val < 99:
#         return 4
#     else:
#         return 5

# for file in glob.glob(os.path.join(input_folder, '*.xlsx')):
#     df = pd.read_excel(file)

#     # Identify columns
#     day_cols = [col for col in df.columns if col.startswith("Day_")]
#     id_cols = ['FID', 'Lat', 'Long']

#     class_df = df[day_cols].map(classify_rainfall)

#     dry_days = (class_df == 1).sum(axis=1)
#     wet_days = (class_df != 1).sum(axis=1)

#     # Combine id and classification
#     full_result = pd.concat([df[id_cols], class_df], axis=1)
#     full_result['Dry_days'] = dry_days
#     full_result['Wet_days'] = wet_days

#     # Add class-wise counts (this is your required addition)
#     for i in range(1, 6):
#         full_result[f'Class_{i}_days'] = (class_df == i).sum(axis=1)

#     # Save
#     base = os.path.basename(file)
#     year = base.split('uk')[0]
#     output_file = os.path.join(output_folder, f'{year}_full_classified.csv')
#     full_result.to_csv(output_file, index=False)
#     print(f"Saved detailed classification for {year} to {output_file}")

# import pandas as pd
# import numpy as np

# df = pd.read_excel('data/1980uk.xlsx')

# # ----- Adjust these day ranges if needed! -----
# season_days = {
#     'Winter':   list(range(1, 60)),      # Day_1 to Day_59 (Jan, Feb)
#     'Summer':   list(range(60, 152)),    # Day_60 to Day_151 (Mar-May)
#     'Monsoon':  list(range(152, 274)),   # Day_152 to Day_273 (Jun-Sep)
#     'Autumn':   list(range(274, 366)),   # Day_274 to Day_365 (Oct-Dec)
# }

# # Create column names per season using Day_XXX format
# season_cols = {season: [f'Day_{d}' for d in day_nums] for season, day_nums in season_days.items()}

# # Function for rainfall classification
# def classify_rainfall(x):
#     if x == 0:
#         return 1
#     elif 0 < x < 10:
#         return 2
#     elif 10 <= x < 50:
#         return 3
#     elif 50 <= x < 99:
#         return 4
#     else:
#         return 5

# output = df[['FID', 'Lat', 'Long']].copy()

# # For each season, calculate
# for season, columns in season_cols.items():
#     # Classify rainfall for this season (subset)
#     season_data = df[columns]
#     class_df = season_data.map(classify_rainfall)

#     # Total rainfall (if you want)
#     output[f'{season}_Rainfall_Total'] = season_data.sum(axis=1)
#     # Dry day count
#     output[f'{season}_Dry_days'] = (season_data == 0).sum(axis=1)
#     # Wet day count
#     output[f'{season}_Wet_days'] = (season_data > 0).sum(axis=1)
#     # Count days in each class
#     for c in range(1, 6):
#         output[f'{season}_Class{c}_days'] = (class_df == c).sum(axis=1)

# output.to_csv('seasonal_classification.csv', index=False)

# import pandas as pd
# import glob
# import os

# # Folder structure
# input_folder = 'data/'        # Your yearly files (e.g., 1980uk.xlsx)
# output_folder = 'classified_seasonal/' # Where results will be stored
# os.makedirs(output_folder, exist_ok=True)

# # Define season day ranges for a 365-day year
# season_days = {
#     'Winter':   list(range(1, 60)),      # Day_1 to Day_59 (Jan, Feb)
#     'Summer':   list(range(60, 152)),    # Day_60 to Day_151 (Mar, Apr, May)
#     'Monsoon':  list(range(152, 274)),   # Day_152 to Day_273 (Jun, Jul, Aug, Sep)
#     'Autumn':   list(range(274, 366)),   # Day_274 to Day_365 (Oct, Nov, Dec)
# }

# # Generate column names for each season
# season_cols = {season: [f'Day_{d}' for d in days] for season, days in season_days.items()}

# def classify_rainfall(x):
#     if x == 0:
#         return 1
#     elif 0 < x < 10:
#         return 2
#     elif 10 <= x < 50:
#         return 3
#     elif 50 <= x < 99:
#         return 4
#     else:
#         return 5

# # Process all Excel files in the input folder
# for file in glob.glob(os.path.join(input_folder, '*.xlsx')):
#     df = pd.read_excel(file)
#     id_cols = ['FID', 'Lat', 'Long']

#     # Output dataframe with node info
#     output = df[id_cols].copy()

#     for season, columns in season_cols.items():
#         # Fall back if some day columns are missing for a file
#         valid_cols = [col for col in columns if col in df.columns]

#         # Seasonal original values and class values
#         season_data = df[valid_cols]
#         class_df = season_data.map(classify_rainfall)

#         # Sum rainfall, dry/wet, and class-wise counts for this season
#         output[f'{season}_Rainfall_Total'] = season_data.sum(axis=1)
#         output[f'{season}_Dry_days'] = (season_data == 0).sum(axis=1)
#         output[f'{season}_Wet_days'] = (season_data > 0).sum(axis=1)
#         for c in range(1, 6):
#             output[f'{season}_Class{c}_days'] = (class_df == c).sum(axis=1)

#     # Save per-year file
#     year = os.path.basename(file).split('uk')[0]
#     output_file = os.path.join(output_folder, f'{year}_seasonal_classified.csv')
#     output.to_csv(output_file, index=False)
#     print(f"Processed {file} -> Saved seasonal statistics to {output_file}")


import pandas as pd
import glob
import os

input_folder = 'data/'                      # Adjust if your folder is different
output_file = 'annual_totals_by_node.csv'   # Output will have all years & nodes

records = []

# # Adjust file glob as needed for xlsx or csv
# for fname in glob.glob(os.path.join(input_folder, '*.xlsx')):
#     # Extract year from filename, e.g., '1980uk.xlsx' -> '1980'
#     basename = os.path.basename(fname)
#     # This extracts the first 4-digit number it finds (assumes year at start of name)
#     import re
#     found = re.search(r'\d{4}', basename)
#     year = int(found.group()) if found else 'Unknown'
    
#     # Read the year file
#     df = pd.read_excel(fname)
    
#     # Identify columns that are Day columns (handles any day range)
#     day_cols = [col for col in df.columns if col.startswith('Day_')]
#     if not day_cols:
#         print(f"No Day_ columns found in {fname}, skipping.")
#         continue

#     # Calculate annual total for each node
#     annual_totals = df[day_cols].sum(axis=1)
#     # Use Lat/Long as the unique node identifier (adjust if needed)
#     nodes = list(zip(df['Lat'], df['Long']))

#     for node, total in zip(nodes, annual_totals):
#         records.append({
#             'Year': year,
#             'Node': str(node),  # or use separate Lat/Long columns if preferred
#             'Annual_Rainfall': total
#         })

# # Save all node-year summaries to CSV
# if records:
#     summary_df = pd.DataFrame(records)
#     summary_df.to_csv(output_file, index=False)
#     print(f"Saved file: {output_file} with {len(summary_df)} rows.")
# else:
#     print("No annual totals were calculated. Please check your folder and column names.")



import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

df = pd.read_csv('annual_totals_by_node.csv')  # Summarized file: columns ['Year', 'Node', 'Annual_Rainfall']

years = sorted(df['Year'].unique())
annual_avg = df.groupby('Year')['Annual_Rainfall'].mean()

# Linear regression for trend
slope, intercept, r_value, p_value, std_err = linregress(years, annual_avg)

plt.plot(years, annual_avg, marker='o', label='Mean Annual Rainfall')
plt.plot(years, intercept + slope * pd.Series(years), 'r--', label='Trend Line')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.title('Annual Rainfall Trend')
plt.legend()
plt.show()
