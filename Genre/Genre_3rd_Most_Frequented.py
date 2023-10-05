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

third_most_frequent_genre = genre_counts.index[2]

# Filter the dataset for the third most frequent genre
genre_data = df[df['Genre'] == third_most_frequent_genre]

# Group and aggregate data at the yearly level
grouped = genre_data.groupby('Year').size().reset_index(name='Count')

# Plot the graph of genre frequency over the years
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Count'], label=third_most_frequent_genre)
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Genre Count Over the Years - Third Top Genre: '+ third_most_frequent_genre)
plt.legend()
plt.show()
