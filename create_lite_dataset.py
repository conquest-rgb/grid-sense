import pandas as pd
import os

print("Current directory:", os.getcwd())

# Load the file
df = pd.read_csv('Data/final_df.csv')

print(f"Original: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Convert hour to datetime
df['hour'] = pd.to_datetime(df['hour'])

# Keep only last 30 days
cutoff_date = df['hour'].max() - pd.Timedelta(days=30)
df_lite = df[df['hour'] >= cutoff_date].copy()

# Optimize data types
for col in df_lite.select_dtypes(include=['float64']).columns:
    df_lite[col] = df_lite[col].astype('float32')

for col in df_lite.select_dtypes(include=['int64']).columns:
    df_lite[col] = df_lite[col].astype('int32')

print(f"Lite: {len(df_lite):,} rows, {df_lite.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Save
df_lite.to_csv('Data/final_df_lite.csv', index=False)