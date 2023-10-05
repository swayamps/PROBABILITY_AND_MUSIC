import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv("billboardHot100_1999-2019.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

df['Genre'] = df['Genre'].str.split(',')

# Explode the 'Genre' column to separate rows for each genre
df = df.explode('Genre')

# Prompt the user to enter the names of the genres
genre1_name = input("Enter the name of the first genre: ")
genre2_name = input("Enter the name of the second genre: ")

# Capitalize the first letter of each word in the genre names
genre1_name_processed = genre1_name.title()
genre2_name_processed = genre2_name.title()

# Filter the dataset for the specified genres
genre1_data = df[df['Genre'].str.lower()==genre1_name_processed.lower()]
genre2_data = df[df['Genre'].str.lower()==genre2_name_processed.lower()]

if genre1_data.empty or genre2_data.empty:
    print("One or both of the genres are not found in the dataset. Please try again.")
else:
    # Group and aggregate data at the yearly level for the specified genres
    genre1_grouped = genre1_data.groupby('Year').size().reset_index(name='Count')
    genre2_grouped = genre2_data.groupby('Year').size().reset_index(name='Count')

    # Plot the graph for the specified genres
    plt.figure(figsize=(10, 6))
    plt.plot(genre1_grouped['Year'], genre1_grouped['Count'], label=genre1_name_processed)
    plt.plot(genre2_grouped['Year'], genre2_grouped['Count'], label=genre2_name_processed)

    plt.xlabel('Year')
    plt.ylabel('Genre Count')
    plt.title('Genre Count Over the Years')
    plt.legend()
    plt.show()