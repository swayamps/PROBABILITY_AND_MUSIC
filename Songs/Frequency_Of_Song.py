import pandas as pd

# Load the dataset
df=pd.read_csv("C:/Users/swaya/OneDrive/Desktop/billboardHot100_1999-2019.csv/billboardHot100_1999-2019.csv") # Replace "your_dataset.csv" with the path to your dataset

# Prompt the user to enter the value to search for
user_input = input("Enter the name of the song: ")

user_input= user_input.title()

# Count the frequency of the user input in a specific column
frequency = df['Name'].value_counts().get(user_input, 0)

print(f"The frequency of '{user_input}' in the dataset is: {frequency}")