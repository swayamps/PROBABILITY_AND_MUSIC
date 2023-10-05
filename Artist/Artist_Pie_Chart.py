import pandas as pd

# Load and preprocess the dataset
df = pd.read_csv("charts.csv")
df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Calculate the frequency of each artist
artist_counts = df['Artists'].value_counts()

# Select top 10 artists and group the rest as "Others"
top_artists = artist_counts.head(10)
other_count = artist_counts[10:].sum()

# Print the top 10 artists
for artist, count in top_artists.items():
    percentage = (count / (count + other_count)) * 100
    print(f"{artist}: {percentage:.2f}%")

# Print the "Others" group
percentage_others = (other_count / (top_artists.sum() + other_count)) * 100
print(f"Others: {percentage_others:.2f}%")