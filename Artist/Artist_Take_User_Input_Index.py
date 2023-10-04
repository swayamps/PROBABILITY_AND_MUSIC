import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Display the available artists
artist_counts = df['Artists'].value_counts()


# Prompt the user to enter the index of the artist
artist_index = int(input("Enter the index of the artist you want to plot: ")) - 1

if artist_index < 0 or artist_index >= len(artist_counts):
    print("Invalid artist index. Please try again.")
else:
    # Get the selected artist based on the index
    selected_artist = artist_counts.index[artist_index]

    # Filter the dataset for the selected artist
    artist_data = df[df['Artists'] == selected_artist]

    # Group and aggregate data at the yearly level for the selected artist
    grouped = artist_data.groupby('Year').size().reset_index(name='Count')

    # Plot the graph for the selected artist
    plt.figure(figsize=(10, 6))
    plt.plot(grouped['Year'], grouped['Count'], label=selected_artist)

    plt.xlabel('Year')
    plt.ylabel('Artist Count')
    plt.title('Artist Count Over the Years - Artist: ' + selected_artist)
    plt.legend()
    plt.show()