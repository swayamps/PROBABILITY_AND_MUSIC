import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/billboardHot100_1999-2019.csv")
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Split the genre column into a list of genres
df['Genre'] = df['Genre'].str.split(',')

# Explode the 'Genre' column to separate rows for each genre
df = df.explode('Genre')

# Count the frequency of each genre
genre_counts = df['Genre'].value_counts()

# Extract the top two genres
top_two_genres = genre_counts.index[:2]

# Filter the dataset for the top two genres
genre_data = df[df['Genre'].isin(top_two_genres)]

# Group and aggregate data at the yearly level for the top two genres
grouped = genre_data.groupby(['Year', 'Genre']).size().reset_index(name='Count')

# Plot the graph for the top two genres
plt.figure(figsize=(10, 6))
for genre in top_two_genres:
    genre_data = grouped[grouped['Genre'] == genre]
    plt.plot(genre_data['Year'], genre_data['Count'], label=genre)

plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Comparison of Top Two Genres Over the Years')
plt.legend()
plt.show()