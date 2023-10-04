import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# Prompt the user to enter the names of the artists
artist1_name = input("Enter the name of the first artist: ")
artist2_name = input("Enter the name of the second artist: ")

# Capitalize the first letter of each word in the artist names
artist1_name_processed = artist1_name.title()
artist2_name_processed = artist2_name.title()

# Filter the dataset for the specified artists
artist1_data = df[df['Artists'].str.lower() == artist1_name_processed.lower()]
artist2_data = df[df['Artists'].str.lower() == artist2_name_processed.lower()]

if artist1_data.empty or artist2_data.empty:
    print("One or both of the artists are not found in the dataset. Please try again.")
else:
    # Group and aggregate data at the yearly level for the specified artists
    artist1_grouped = artist1_data.groupby('Year').size().reset_index(name='Count')
    artist2_grouped = artist2_data.groupby('Year').size().reset_index(name='Count')

    # Plot the graph for the specified artists
    plt.figure(figsize=(10, 6))
    plt.plot(artist1_grouped['Year'], artist1_grouped['Count'], label=artist1_name_processed)
    plt.plot(artist2_grouped['Year'], artist2_grouped['Count'], label=artist2_name_processed)

    plt.xlabel('Year')
    plt.ylabel('Artist Count')
    plt.title('Artist Count Over the Years')
    plt.legend()
    plt.show()