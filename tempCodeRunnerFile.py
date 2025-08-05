import pandas as pd
# import numpy as np
# import glob
# import os
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVR
# import time

# # Path to your data directory (UPDATE THIS to your local path)
# data_folder = 'PUT_YOUR_LOCAL_PATH_HERE'

# # Find all relevant Excel files
# all_files = glob.glob(os.path.join(data_folder, '*uk.xlsx'))
# all_files.sort()  # Ensures chronological order

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
#     k = 80
#     coords = melted[['Lat', 'Long']].values
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(coords)

#     melted['Cluster'] = kmeans.labels_
#     clustered = melted.groupby('Cluster').mean(numeric_only=True).reset_index()

#     dfs.append(clustered)

# # Concatenate all years into a single DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)
# combined_df.to_csv('rainfall_all_years_long.csv', index=False)
# total_rows = combined_df.shape[0]
# print("Total number of rows:", total_rows)

# # Features to use for prediction
# features = ['FID', 'Lat', 'Long', 'DAYNO.', 'YEAR']
# X = combined_df[features]
# y = combined_df['RAINFALL']

# # Dictionary of models
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

# # List of test split sizes (test size, train size)
# splits = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

# for test_size, train_size in splits:
#     print(f"\n--- Train/Test Split: {int(train_size*100)}% train / {int(test_size*100)}% test ---")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42
#     )

#     for name, model in models.items():
#         start_time = time.time()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         end_time = time.time()
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = mean_squared_error(y_test, y_pred, squared=False)  # Use squared=False for RMSE
#         r2 = r2_score(y_test, y_pred)
#         elapsed = end_time - start_time
#         print(f'{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Time={elapsed:.2f}s')

