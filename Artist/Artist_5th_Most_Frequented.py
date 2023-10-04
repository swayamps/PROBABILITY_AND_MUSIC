import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Calculate the frequency of each artist
artist_counts = df['Artists'].value_counts()

# Get the artist with the fifth-highest frequency
fifth_top_artist = artist_counts.index[4]

# Filter the dataset for the fifth top artist
fifth_top_artist_data = df[df['Artists'] == fifth_top_artist]

# Group and aggregate data at the yearly level for the fifth top artist
grouped = fifth_top_artist_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph for the fifth top artist
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=fifth_top_artist)

plt.xlabel('Year')
plt.ylabel('Artist Count')
plt.title('Artist Count Over the Years - Fifth Top Artist: ' + fifth_top_artist)
plt.legend()
plt.show()