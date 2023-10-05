import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv("billboardHot100_1999-2019.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

df['Genre'] = df['Genre'].str.split(',')

# Explode the 'Genre' column to separate rows for each genre
df = df.explode('Genre')

# Display the available genres
genre_counts = df['Genre'].value_counts()


# Prompt the user to enter the index of the genre
genre_index = int(input("Enter the index of the genre you want to plot: ")) - 1

if genre_index < 0 or genre_index >= len(genre_counts):
    print("Invalid genre index. Please try again.")
else:
    # Get the selected genre based on the index
    selected_genre = genre_counts.index[genre_index]

    # Filter the dataset for the selected genre
    genre_data = df[df['Genre'] == selected_genre]

    # Group and aggregate data at the yearly level for the selected genre
    grouped = genre_data.groupby('Year').size().reset_index(name='Count')

    # Plot the graph for the selected genre
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['Year'], grouped['Count'], label=selected_genre)

    plt.xlabel('Year')
    plt.ylabel('Genre Count')
    plt.title('Genre Count Over the Years - Genre: ' + selected_genre)
    plt.legend()
    plt.show()