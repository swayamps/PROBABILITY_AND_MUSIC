import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
artist_counts = df['Artists'].value_counts()

# Get the artist with the highest frequency
top_artist = artist_counts.idxmax()

# Filter the dataset for the top artist
top_artist_data = df[df['Artists'] == top_artist]

# Group and aggregate data at the yearly level for the top artist
grouped = top_artist_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph for the top artist
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=top_artist)

plt.xlabel('Year')
plt.ylabel('Artist Count')
plt.title('Artist Count Over the Years - Top Artist: ' + top_artist)
plt.legend()
plt.show()

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
artist_counts = df['Artists'].value_counts()

# Get the artist with the highest frequency
top_artist = artist_counts.idxmax()

# Filter the dataset for the top artist
top_artist_data = df[df['Artists'] == top_artist]

# Group and aggregate data at the yearly level for the top artist
grouped = top_artist_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph for the top artist
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=top_artist)

plt.xlabel('Year')
plt.ylabel('Artist Count')
plt.title('Artist Count Over the Years - Top Artist: ' + top_artist)
plt.legend()
plt.show()