import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Calculate the frequency of each artist
artist_counts = df['Artists'].value_counts()

# Get the artist with the sixth-highest frequency
sixth_top_artist = artist_counts.index[5]

# Filter the dataset for the sixth top artist
sixth_top_artist_data = df[df['Artists'] == sixth_top_artist]

# Group and aggregate data at the yearly level for the sixth top artist
grouped = sixth_top_artist_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph for the sixth top artist
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=sixth_top_artist)

plt.xlabel('Year')
plt.ylabel('Artist Count')
plt.title('Artist Count Over the Years - Sixth Top Artist: ' + sixth_top_artist)
plt.legend()
plt.show()