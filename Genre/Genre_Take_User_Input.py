import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv("billboardHot100_1999-2019.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

df['Genre'] = df['Genre'].str.split(',')

# Explode the 'Genre' column to separate rows for each genre
df = df.explode('Genre')

while True:
    # Prompt the user to enter the genre
    genre_name = input("Enter the name of the genre: ")

    # Capitalize the first letter of each word in the genre name
    genre_name_processed = genre_name.title()



    # Filter the dataset for the specified genre
    genre_data = df[df['Genre'].str.lower()==genre_name_processed.lower()]

    if genre_data.empty:
        print("The genre is not found in the dataset. Please try again.")
    else:
        # Group and aggregate data at the yearly level for the specified genre
        grouped = genre_data.groupby('Year').size().reset_index(name='Count')

        # Plot the graph for the specified genre
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['Year'], grouped['Count'], label=genre_name_processed)

        plt.xlabel('Year')
        plt.ylabel('Genre Count')
        plt.title('Genre Count Over the Years - Genre: ' + genre_name_processed)
        plt.legend()
        plt.show()
        break