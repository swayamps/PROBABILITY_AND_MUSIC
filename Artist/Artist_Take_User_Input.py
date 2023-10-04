import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/charts.csv")

# Convert year column to datetime format
df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

while True:
    # Prompt the user to enter the name of the artist
    artist_name = input("Enter the name of the artist: ")

    # Capitalize the first letter of each word in the artist name
    artist_name_processed = artist_name.title()

    # Filter the dataset for the specified artist
    artist_data = df[df['Artists'].str.lower() == artist_name_processed.lower()]

    if artist_data.empty:
        print("The artist is not found in the dataset. Please try again.")
    else:
        # Group and aggregate data at the yearly level for the specified artist
        grouped = artist_data.groupby('Year').size().reset_index(name='Count')

        # Plot the graph for the specified artist
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['Year'], grouped['Count'], label=artist_name_processed)

        plt.xlabel('Year')
        plt.ylabel('Artist Count')
        plt.title('Artist Count Over the Years - Artist: ' + artist_name_processed)
        plt.legend()
        plt.show()
        break


# while True:
#     # Prompt the user to enter the name of the artist
#     artist_name = input("Enter the name of the artist: ")

#     # Convert the artist name to lowercase
#     artist_name = artist_name.lower()

#     # Filter the dataset for the specified artist
#     artist_data = df[df['Artists'].str.lower() == artist_name]

#     if artist_data.empty:
#         print("The artist is not found in the dataset. Please try again.")
#     else:
#         # Group and aggregate data at the yearly level for the specified artist
#         grouped = artist_data.groupby('Year').size().reset_index(name='Count')

#         # Plot the graph for the specified artist
#         plt.figure(figsize=(10, 6))
#         plt.plot(grouped['Year'], grouped['Count'], label=artist_name)

#         plt.xlabel('Year')
#         plt.ylabel('Artist Count')
#         plt.title('Artist Count Over the Years - Artist: ' + artist_name)
#         plt.legend()
#         plt.show()
#         break

# while True:
#     # Prompt the user to enter the name of the artist
#     artist_name = input("Enter the name of the artist: ")

#     # Filter the dataset for the specified artist
#     artist_data = df[df['Artists'] == artist_name]

#     if artist_data.empty:
#         print("The artist is not found in the dataset. Please try again")
#     else:
#         # Group and aggregate data at the yearly level for the specified artist
#         grouped = artist_data.groupby('Year').size().reset_index(name='Count')

#         # Plot the graph for the specified artist
#         plt.figure(figsize=(10, 6))
#         plt.plot(grouped['Year'], grouped['Count'], label=artist_name)

#         plt.xlabel('Year')
#         plt.ylabel('Artist Count')
#         plt.title('Artist Count Over the Years - Artist: ' + artist_name)
#         plt.legend()
#         plt.show()
#         break



# # Prompt the user to enter the name of the artist
# artist_name = input("Enter the name of the artist: ")

# # Filter the dataset for the specified artist
# artist_data = df[df['Artists'] == artist_name]

# # Group and aggregate data at the yearly level for the specified artist
# grouped = artist_data.groupby('Year').size().reset_index(name='Count')

# # Plot the graph for the specified artist
# plt.figure(figsize=(10, 6))
# plt.plot(grouped['Year'], grouped['Count'], label=artist_name)

# plt.xlabel('Year')
# plt.ylabel('Artist Count')
# plt.title('Artist Count Over the Years - Artist: ' + artist_name)
# plt.legend()
# plt.show()