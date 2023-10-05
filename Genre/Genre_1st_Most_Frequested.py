import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("billboardHot100_1999-2019.csv")

df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Split the genre column into a list of genres
df['Genres'] = df['Genre'].str.split(',')

# Flatten the list of genres
genres_list = [genre for genres in df['Genres'] for genre in genres]

# Calculate the frequency of each genre
genre_counts = pd.Series(genres_list).value_counts()

# Get the genre with the highest frequency
top_genre = genre_counts.idxmax()

# Filter the dataset for the top genre
top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# Group and aggregate data at the yearly level for the top genre
grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph for the top genre
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=top_genre)

plt.xlabel('Year')
plt.ylabel('Genre Count')
plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
plt.legend()
plt.show()
