import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("charts.csv")


# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Calculate the frequency of each artist
artist_counts = df['Artists'].value_counts()

top_artists = artist_counts.index[:2]

# Filter the dataset for the top two artists
top_artists_data = df[df['Artists'].isin(top_artists)]

# Group and aggregate data at the yearly level for the top two artists
grouped = top_artists_data.groupby(['Year', 'Artists']).size().reset_index(name='Count')

# Plot the graph for the top two artistss
plt.figure(figsize=(10, 6))
for artists in top_artists:
    artists_data = grouped[grouped['Artists'] == artists]
    plt.plot(artists_data['Year'], artists_data['Count'], label=artists)

plt.xlabel('Year')
plt.ylabel('Artist Count')
plt.title('Artist Count Over the Years - Top Artists')
plt.legend()
plt.show()