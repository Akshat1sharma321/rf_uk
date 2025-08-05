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


# import pandas as pd
# import glob
# import os

# input_folder = 'data/'                      # Adjust if your folder is different
# output_file = 'annual_totals_by_node.csv'   # Output will have all years & nodes

# records = []

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



# import pandas as pd
# from scipy.stats import linregress
# import matplotlib.pyplot as plt

# df = pd.read_csv('annual_totals_by_node.csv')  # Summarized file: columns ['Year', 'Node', 'Annual_Rainfall']

# years = sorted(df['Year'].unique())
# annual_avg = df.groupby('Year')['Annual_Rainfall'].mean()

# # Linear regression for trend
# slope, intercept, r_value, p_value, std_err = linregress(years, annual_avg)

# plt.plot(years, annual_avg, marker='o', label='Mean Annual Rainfall')
# plt.plot(years, intercept + slope * pd.Series(years), 'r--', label='Trend Line')
# plt.xlabel('Year')
# plt.ylabel('Rainfall (mm)')
# plt.title('Annual Rainfall Trend')
# plt.legend()
# plt.show()
# import pandas as pd
# import glob
# import os
# print("Hello")


# df = pd.read_excel('/data/1980uk.xlsx')

# # Path to your data directory
# data_folder = 'rf_pred\data'

# # Find all relevant Excel files
# all_files = glob.glob(os.path.join(data_folder, '*uk.xlsx'))
# all_files.sort()  # Ensures chronological order
# print(f"Found {len(all_files)} files")
# print(all_files)  # To see file paths

# dfs = []

# for filename in all_files:
#     # Extract the year from the filename, e.g. "1980uk.xlsx" → 1980
#     basename = os.path.basename(filename)
#     year = int(basename[:4])

#     # Read Excel file
#     df = pd.read_excel(filename)

#     # Melt as before
#     melted = df.melt(
#         id_vars=['FID', 'Lat', 'Long'],
#         value_vars=[col for col in df.columns if col.startswith('Day_')],
#         var_name='DAYNO.',
#         value_name='RAINFALL'
#     )
#     melted['DAYNO.'] = melted['DAYNO.'].str.replace('Day_', '').astype(int)
#     melted['YEAR'] = year

#     dfs.append(melted)

# # Concatenate all years into a single DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# # Save combined data if needed
# combined_df.to_csv('rainfall_all_years_long.csv', index=False)

# total_rows = combined_df.shape[0]
# print("Total number of rows:", total_rows)
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Features to use for prediction
# features = ['FID', 'Lat', 'Long', 'DAYNO.', 'YEAR']
# X = combined_df[features]
# y = combined_df['RAINFALL']

# # # Train-test split (e.g., 80% train, 20% test)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # import time
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(),
#     'Lasso Regression': Lasso(),
#     'Decision Tree': DecisionTreeRegressor(),
#     'Random Forest': RandomForestRegressor(),
#     'Gradient Boosting': GradientBoostingRegressor(),
#     'XGBoost': XGBRegressor(),
#     'SVR': SVR(),
#     'KNN': KNeighborsRegressor(),
#     'MLP': MLPRegressor(max_iter=500)
# }

# for name, model in models.items():
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     end_time = time.time()
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred) # Removed squared=False
#     r2 = r2_score(y_test, y_pred)
#     elapsed = end_time - start_time
#     print(f'{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Time={elapsed:.2f}s')

# import time
# from sklearn.linear_model import ElasticNet, HuberRegressor, QuantileRegressor
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # For neural models
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
# from tensorflow.keras.optimizers import Adam

# # Prepare data
# features = ['FID', 'Lat', 'Long', 'DAYNO.', 'YEAR']
# X = combined_df[features]
# y = combined_df['RAINFALL']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("=== Additional Regression Models ===")

# # 1. Elastic Net Regression
# start = time.time()
# elastic = ElasticNet()
# elastic.fit(X_train, y_train)
# y_pred = elastic.predict(X_test)
# end = time.time()
# print(f"ElasticNet: MAE={mean_absolute_error(y_test, y_pred):.3f}  RMSE={mean_squared_error(y_test, y_pred, squared=False):.3f}  R2={r2_score(y_test, y_pred):.3f}  Time={end - start:.2f}s")

# # 2. Partial Least Squares Regression
# start = time.time()
# pls = PLSRegression(n_components=2)
# pls.fit(X_train, y_train)
# y_pred = pls.predict(X_test).flatten()
# end = time.time()
# print(f"PLSRegression: MAE={mean_absolute_error(y_test, y_pred):.3f}  RMSE={mean_squared_error(y_test, y_pred, squared=False):.3f}  R2={r2_score(y_test, y_pred):.3f}  Time={end - start:.2f}s")

# # 3. Huber Regression (robust to outliers)
# start = time.time()
# huber = HuberRegressor()
# huber.fit(X_train, y_train)
# y_pred = huber.predict(X_test)
# end = time.time()
# print(f"Huber: MAE={mean_absolute_error(y_test, y_pred):.3f}  RMSE={mean_squared_error(y_test, y_pred, squared=False):.3f}  R2={r2_score(y_test, y_pred):.3f}  Time={end - start:.2f}s")

# # 4. Quantile Regression (Median regression)
# start = time.time()
# quantile = QuantileRegressor(quantile=0.5, solver='highs')

# quantile.fit(X_train, y_train)
# y_pred = quantile.predict(X_test)
# end = time.time()
# print(f"Quantile (Median): MAE={mean_absolute_error(y_test, y_pred):.3f}  RMSE={mean_squared_error(y_test, y_pred, squared=False):.3f}  R2={r2_score(y_test, y_pred):.3f}  Time={end - start:.2f}s")

# # ========== 1D Convolutional Neural Network (CNN) ==========
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# seq_length = 7

# def create_sequences(X, y, seq_length=7):
#     Xs, ys = [], []
#     for i in range(len(X) - seq_length):
#         Xs.append(X[i:i+seq_length])
#         ys.append(y[i+seq_length])
#     return np.array(Xs), np.array(ys)

# X_seq, y_seq = create_sequences(X_scaled, y.values, seq_length=seq_length)
# X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# cnn_model = Sequential([
#     Conv1D(32, 3, activation='relu', input_shape=(seq_length, X_seq.shape[2])),
#     Dropout(0.2),
#     Flatten(),
#     Dense(32, activation='relu'),
#     Dense(1)
# ])
# cnn_model.compile(optimizer=Adam(), loss='mae')

# start = time.time()
# cnn_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, verbose=0)
# end = time.time()
# y_pred = cnn_model.predict(X_test_seq).flatten()
# print(f"CNN: MAE={mean_absolute_error(y_test_seq, y_pred):.3f}  RMSE={mean_squared_error(y_test_seq, y_pred, squared=False):.3f}  R2={r2_score(y_test_seq, y_pred):.3f}  Time={end - start:.2f}s")

# # ========== Transformer-Based Model ==========

# # Simple transformer block
# class SimpleTransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ln1 = LayerNormalization()
#         self.ffn = Sequential([Dense(embed_dim, activation="relu"), Dense(embed_dim)])
#         self.ln2 = LayerNormalization()
#     def call(self, x):
#         attn = self.att(x, x)
#         out1 = self.ln1(x + attn)
#         ffn_out = self.ffn(out1)
#         return self.ln2(out1 + ffn_out)

# import tensorflow as tf

# embed_dim = X_seq.shape[2]
# num_heads = 2
# inputs = Input(shape=(seq_length, embed_dim))
# x = SimpleTransformerBlock(embed_dim, num_heads)(inputs)
# x = GlobalAveragePooling1D()(x)
# outputs = Dense(1)(x)

# transformer_model = tf.keras.Model(inputs, outputs)
# transformer_model.compile(optimizer='adam', loss='mae')

# start = time.time()
# transformer_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=64, verbose=0)
# end = time.time()
# y_pred = transformer_model.predict(X_test_seq).flatten()
# print(f"Transformer: MAE={mean_absolute_error(y_test_seq, y_pred):.3f}  RMSE={mean_squared_error(y_test_seq, y_pred, squared=False):.3f}  R2={r2_score(y_test_seq, y_pred):.3f}  Time={end - start:.2f}s")
# pip install pandas numpy scikit-learn xgboost openpyxl
# data_folder = "data"
# data_folder = r"C:\Users\AJAY1\OneDrive\Desktop\Research\rf_pred\data"
data_folder = 'rf_pred\data'
print("Hello")
import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import time

# Path to your data directory (UPDATE THIS to your local path)
# data_folder = 'RF_PRED\\data'
data_folder = 'data'
print("Hi")
# Find all relevant Excel files
all_files = glob.glob(os.path.join(data_folder, '*uk.xlsx'))
print("Files found:", all_files)

all_files.sort()  # Ensures chronological order

dfs = []
print("Hi")
for filename in all_files:
    # Extract the year from the filename, e.g. "1980uk.xlsx" → 1980
    basename = os.path.basename(filename)
    year = int(basename[:4])

    # Read Excel file
    df = pd.read_excel(filename)

    # Melt as before
    melted = df.melt(
        id_vars=['FID', 'Lat', 'Long'],
        value_vars=[col for col in df.columns if col.startswith('Day_')],
        var_name='DAYNO.',
        value_name='RAINFALL'
    )
    melted['DAYNO.'] = melted['DAYNO.'].str.replace('Day_', '').astype(int)
    melted['YEAR'] = year
    k = 80
    coords = melted[['Lat', 'Long']].values
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coords)

    melted['Cluster'] = kmeans.labels_
    clustered = melted.groupby('Cluster').mean(numeric_only=True).reset_index()

    dfs.append(clustered)
print("Hi")
# Concatenate all years into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv('rainfall_all_years_long.csv', index=False)
total_rows = combined_df.shape[0]
print("Total number of rows:", total_rows)

# Features to use for prediction
features = ['FID', 'Lat', 'Long', 'DAYNO.', 'YEAR']
X = combined_df[features]
y = combined_df['RAINFALL']

# Dictionary of models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(max_iter=500)
}

# List of test split sizes (test size, train size)
splits = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

for test_size, train_size in splits:
    print(f"\n--- Train/Test Split: {int(train_size*100)}% train / {int(test_size*100)}% test ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        mae = mean_absolute_error(y_test, y_pred)
        # rmse = mean_squared_error(y_test, y_pred, squared=False) 
         # Use squared=False for RMSE
        # import numpy as npgit 
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2 = r2_score(y_test, y_pred)
        elapsed = end_time - start_time
        print(f'{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Time={elapsed:.2f}s')

