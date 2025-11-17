import pandas as pd

# Load dataset
df = pd.read_csv("Trial_Data.csv")

# Keep only the desired columns
df_reduced = df[['lang', 'id', 'text', 'polarization']]

# Save to a new CSV if needed
df_reduced.to_csv("Trail_Data.csv", index=False)

print(df_reduced.head())
