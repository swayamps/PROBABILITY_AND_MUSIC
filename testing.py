# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df=pd.read_csv("billboardHot100_1999-2019.csv")


# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')


# # Split the genre column into a list of genres
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# second_most_frequent_genre = genre_counts.index[1]

# # Filter the dataset for the second most frequent genre
# genre_data = df[df['Genre'] == second_most_frequent_genre]

# # Group and aggregate data at the yearly level
# grouped = genre_data.groupby('Year').size().reset_index(name='Count')

# # Plot the graph of genre frequency over the years
# plt.figure(figsize=(10, 6))
# plt.plot(grouped['Year'], grouped['Count'], label=second_most_frequent_genre)
# plt.xlabel('Year')
# plt.ylabel('Frequency')
# plt.title('Frequency of Second Most Frequent Genre Over the Years: '+ second_most_frequent_genre)
# plt.legend()
# plt.show()

# # # Calculate the frequency of each genre
# # genre_counts = df['Genre'].value_counts()

# # # Get the genre with the highest frequency
# # top_genre = genre_counts.idxmax()

# # # Filter the dataset for the top genre
# # top_genre_data = df[df['Genre'] == top_genre]

# # # Group and aggregate data at the yearly level for the top genre
# # grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # # Plot the graph for the top genre
# # plt.figure(figsize=(10, 6))
# # plt.plot(grouped['Year'], grouped['Count'], label=top_genre)

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
# # plt.legend()
# # plt.show()


# # # Split the genre column into a list of genres
# # df['Genres'] = df['Genre'].str.split(',')

# # # Flatten the list of genres
# # genres_list = [genre for genres in df['Genres'] for genre in genres]

# # # Calculate the frequency of each genre
# # genre_counts = pd.Series(genres_list).value_counts()

# # # Get the genre with the highest frequency
# # top_genre = genre_counts.idxmax()

# # # Filter the dataset for the top genre
# # top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # # Group and aggregate data at the yearly level for the top genre
# # grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # # Plot the graph for the top genre
# # plt.figure(figsize=(10, 6))
# # plt.plot(grouped['Year'], grouped['Count'], label=top_genre)

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
# # plt.legend()
# # plt.show()

# # # Split the genres by comma and convert them to a list
# # df['Genre'] = df['Genre'].str.split(',')

# # # Explode the list of genres into separate rows
# # df = df.explode('Genre')

# # # Group and aggregate data
# # grouped = df.groupby(['Year', 'Genre']).size().reset_index(name='Count')

# # # Determine the topmost genre for each year
# # top_genre_per_year = grouped.groupby('Year')['Count'].idxmax()
# # top_genre_data = grouped.loc[top_genre_per_year]

# # # Plot the graph for the topmost genre
# # plt.figure(figsize=(10, 6))
# # plt.plot(top_genre_data['Year'], top_genre_data['Count'], label=top_genre_data['Genre'].values[0])

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Topmost Genre and Genre Count Over the Years')
# # plt.legend()
# # plt.show()






# # # Group and aggregate data
# # grouped = df.groupby(['Year', 'Genre']).size().reset_index(name='Count')

# # # Determine the topmost genre for each year
# # top_genre_per_year = grouped.groupby('Year')['Count'].idxmax()
# # top_genre_data = grouped.loc[top_genre_per_year]

# # # Plot the graph for the topmost genre
# # plt.figure(figsize=(10, 6))
# # plt.plot(top_genre_data['Year'], top_genre_data['Count'], label=top_genre_data['Genre'].values[0])

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Topmost Genre and Genre Count Over the Years')
# # plt.legend()
# # plt.show()
# # print(top_genre_data['Count'])









# # # Rename the 'Genre' column
# # df.rename(columns={'Genre': 'Genre_Count'}, inplace=True)

# # # Group and aggregate data
# # grouped = df.groupby(['Year', 'Genre_Count']).size().reset_index(name='Count')

# # # Determine most frequent genres
# # most_frequent_genres = grouped.groupby('Year')['Count'].idxmax()
# # most_frequent_genres_df = grouped.loc[most_frequent_genres]

# # # Plot the graph
# # plt.figure(figsize=(10, 6))
# # for genre in most_frequent_genres_df['Genre_Count'].unique():
# #     genre_data = most_frequent_genres_df[most_frequent_genres_df['Genre_Count'] == genre]
# #     plt.plot(genre_data['Year'], genre_data['Count'], label=genre)

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Most Frequent Genres and Genre Count Over the Years')
# # plt.legend()
# # plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")

# # Convert year column to datetime format
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# while True:
#     # Prompt the user to enter the genre
#     genre_name = input("Enter the name of the genre: ")

#     # Capitalize the first letter of each word in the genre name
#     genre_name_processed = genre_name.title()

#     df['Genre'] = df['Genre'].str.split(',')

#     # Explode the 'Genre' column to separate rows for each genre
#     df = df.explode('Genre')

#     # Filter the dataset for the specified genre
#     genre_data = df[df['Genre'].str.contains(genre_name_processed, case=False)]

#     if genre_data.empty:
#         print("The genre is not found in the dataset. Please try again.")
#     else:
#         # Group and aggregate data at the yearly level for the specified genre
#         grouped = genre_data.groupby('Year').size().reset_index(name='Count')

#         # Plot the graph for the specified genre
#         plt.figure(figsize=(10, 6))
#         plt.plot(grouped['Year'], grouped['Count'], label=genre_name_processed)

#         plt.xlabel('Year')
#         plt.ylabel('Genre Count')
#         plt.title('Genre Count Over the Years - Genre: ' + genre_name_processed)
#         plt.legend()
#         plt.show()
#         break

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")

# # Convert year column to datetime format
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# df['Genre'] = df['Genre'].str.split(',')

# # Explode the 'Genre' column to separate rows for each genre
# df = df.explode('Genre')

# # Prompt the user to enter the names of the genres
# genre1_name = input("Enter the name of the first genre: ")
# genre2_name = input("Enter the name of the second genre: ")

# # Capitalize the first letter of each word in the genre names
# genre1_name_processed = genre1_name.title()
# genre2_name_processed = genre2_name.title()

# # Filter the dataset for the specified genres
# genre1_data = df[df['Genre'].str.contains(genre1_name_processed, case=False)]
# genre2_data = df[df['Genre'].str.contains(genre2_name_processed, case=False)]

# if genre1_data.empty or genre2_data.empty:
#     print("One or both of the genres are not found in the dataset. Please try again.")
# else:
#     # Group and aggregate data at the yearly level for the specified genres
#     genre1_grouped = genre1_data.groupby('Year').size().reset_index(name='Count')
#     genre2_grouped = genre2_data.groupby('Year').size().reset_index(name='Count')

#     # Plot the graph for the specified genres
#     plt.figure(figsize=(10, 6))
#     plt.plot(genre1_grouped['Year'], genre1_grouped['Count'], label=genre1_name_processed)
#     plt.plot(genre2_grouped['Year'], genre2_grouped['Count'], label=genre2_name_processed)

#     plt.xlabel('Year')
#     plt.ylabel('Genre Count')
#     plt.title('Genre Count Over the Years')
#     plt.legend()
#     plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")

# # Convert year column to datetime format
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# df['Genre'] = df['Genre'].str.split(',')

# # Explode the 'Genre' column to separate rows for each genre
# df = df.explode('Genre')

# # Display the available genres
# genre_counts = df['Genre'].value_counts()


# # Prompt the user to enter the index of the genre
# genre_index = int(input("Enter the index of the genre you want to plot: ")) - 1

# if genre_index < 0 or genre_index >= len(genre_counts):
#     print("Invalid genre index. Please try again.")
# else:
#     # Get the selected genre based on the index
#     selected_genre = genre_counts.index[genre_index]

#     # Filter the dataset for the selected genre
#     genre_data = df[df['Genre'] == selected_genre]

#     # Group and aggregate data at the yearly level for the selected genre
#     grouped = genre_data.groupby('Year').size().reset_index(name='Count')

#     # Plot the graph for the selected genre
#     plt.figure(figsize=(10, 6))
#     plt.plot(grouped['Year'], grouped['Count'], label=selected_genre)

#     plt.xlabel('Year')
#     plt.ylabel('Genre Count')
#     plt.title('Genre Count Over the Years - Genre: ' + selected_genre)
#     plt.legend()
#     plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df=pd.read_csv("billboardHot100_1999-2019.csv")

# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# # Split the genre column into a list of genres
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the yearly level for the top genre
# grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # # Plot the graph for the top genre
# # plt.figure(figsize=(10, 6))
# # plt.plot(grouped['Year'], grouped['Count'], label=top_genre)

# # plt.xlabel('Year')
# # plt.ylabel('Genre Count')
# # plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
# # plt.legend()
# # plt.show()
# from sklearn import linear_model

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(grouped['Year'], grouped['Count'],test_size=0.2,random_state=0)

# reg = linear_model.LinearRegression()
# reg.fit(x_train,y_train)

# reg.coef_

# reg.intercept_

# y_pedo=reg.predict(x_test)

# x_test['G3'] = y_pedo

# from sklearn.metrics import r2_score
# score = r2_score(y_test,y_pedo)
# print("The accuracy of my model is {}%".format(round(score,2)*100))

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the yearly level for the top genre
# grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Year'].dt.year, grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Create and fit the linear regression model
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = reg.predict(x_test)

# x_test['Predicted'] = y_pred

# print(x_test)

# # Calculate the R-squared score
# score = r2_score(y_test, y_pred)
# print("The accuracy of my model is {}%".format(round(score * 100, 2)))

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the yearly level for the top genre
# grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Year'].dt.year, grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Create and fit the linear regression model
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = reg.predict(x_test)

# # Calculate the R-squared score
# score = r2_score(y_test, y_pred)
# print("The accuracy of my model is {}%".format(round(score * 100, 2)))

# # Plot the graph comparing the predicted and actual values
# plt.figure(figsize=(10, 6))
# plt.plot(x_test, y_test, color='blue', label='Actual')
# plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
# plt.xlabel('Year')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')
# df = df.explode('Genre')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the yearly level for the top genre
# grouped = top_genre_data.groupby('Year').size().reset_index(name='Count')

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Year'].dt.year, grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Create and fit the linear regression model
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = reg.predict(x_test)

# # Calculate the R-squared score
# score = r2_score(y_test, y_pred)
# print("The accuracy of my model is {}%".format(round(score * 100, 2)))

# # Plot the graph comparing the predicted and actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(x_test, y_test, color='blue', label='Actual')
# plt.scatter(x_test, y_pred, color='red', linewidth=2, label='Predicted')
# plt.xlabel('Year')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Years - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Create and fit the linear regression model
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = reg.predict(x_test)

# # Calculate the R-squared score
# score = r2_score(y_test, y_pred)
# print("The accuracy of my model is {}%".format(round(score * 100, 2)))

# # Plot the simplified graph comparing the predicted and actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(x_test, y_test, color='blue', label='Actual')
# plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# df = df.explode('Genre')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre =genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.9, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Create and fit the linear regression model
# reg = linear_model.LinearRegression()
# reg.fit(x_train, y_train)

# # Make predictions on the test set
# y_pred = reg.predict(x_test)

# # Calculate the R-squared score
# score = r2_score(y_test, y_pred)
# print("The accuracy of my model is {}%".format(round(score * 100, 2)))

# # Plot the simplified graph comparing the predicted and actual values
# plt.figure(figsize=(10, 6))
# plt.plot(grouped['Week'], grouped['Count'], color='blue', label='Actual')
# plt.plot(grouped['Week'], reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Predicted')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()

#_________________________________________________________________________________________________________________________________________________________________________________________


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Linear Regression
# linear_reg = LinearRegression()
# linear_reg.fit(x_train, y_train)
# linear_pred = linear_reg.predict(x_test)
# linear_score = r2_score(y_test, linear_pred)

# # Decision Tree Regression
# dt_reg = DecisionTreeRegressor(random_state=0)
# dt_reg.fit(x_train, y_train)
# dt_pred = dt_reg.predict(x_test)
# dt_score = r2_score(y_test, dt_pred)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)
# rf_score = r2_score(y_test, rf_pred)

# # Print the accuracy scores
# print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
# print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
# print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))

# # Plot the simplified graph comparing the predicted and actual values
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], color='blue', label='Actual')
# #plt.plot(grouped['Week'], linear_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Linear Regression')
# plt.plot(grouped['Week'], dt_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Decision Tree Regression')
# #plt.plot(grouped['Week'], rf_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='orange', linewidth=2, label='Random Forest Regression')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()

#_______________________________________________________________________________________________________________________________________________



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# # ...

# # Split the data into training and test sets
# # ...

# # Linear Regression
# linear_reg = LinearRegression()
# linear_reg.fit(x_train, y_train)
# linear_pred = linear_reg.predict(x_test)
# linear_score = r2_score(y_test, linear_pred)

# # Ridge Regression
# ridge_reg = Ridge(alpha=1.0)
# ridge_reg.fit(x_train, y_train)
# ridge_pred = ridge_reg.predict(x_test)
# ridge_score = r2_score(y_test, ridge_pred)

# # Lasso Regression
# lasso_reg = Lasso(alpha=1.0)
# lasso_reg.fit(x_train, y_train)
# lasso_pred = lasso_reg.predict(x_test)
# lasso_score = r2_score(y_test, lasso_pred)

# # ElasticNet Regression
# elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
# elastic_net_reg.fit(x_train, y_train)
# elastic_net_pred = elastic_net_reg.predict(x_test)
# elastic_net_score = r2_score(y_test, elastic_net_pred)

# # Decision Tree Regression
# dt_reg = DecisionTreeRegressor(random_state=0)
# dt_reg.fit(x_train, y_train)
# dt_pred = dt_reg.predict(x_test)
# dt_score = r2_score(y_test, dt_pred)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)
# rf_score = r2_score(y_test, rf_pred)

# # Gradient Boosting Regression
# gb_reg = GradientBoostingRegressor(random_state=0)
# gb_reg.fit(x_train, y_train)
# gb_pred = gb_reg.predict(x_test)
# gb_score = r2_score(y_test, gb_pred)

# # Support Vector Regression
# svr_reg = SVR(kernel='rbf')
# svr_reg.fit(x_train, y_train)
# svr_pred = svr_reg.predict(x_test)
# svr_score = r2_score(y_test, svr_pred)

# # Print the accuracy scores
# print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
# print("Ridge Regression Accuracy: {:.2f}%".format(ridge_score * 100))
# print("Lasso Regression Accuracy: {:.2f}%".format(lasso_score * 100))
# print("ElasticNet Regression Accuracy: {:.2f}%".format(elastic_net_score * 100))
# print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
# print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))
# print("Gradient Boosting Regression Accuracy: {:.2f}%".format(gb_score * 100))
# print("Support Vector Regression Accuracy: {:.2f}%".format(svr_score * 100))

# # Plot the simplified graph comparing the predicted and actual values
# # ...

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Linear Regression
# linear_reg = LinearRegression()
# linear_reg.fit(x_train, y_train)
# linear_pred = linear_reg.predict(x_test)
# linear_score = r2_score(y_test, linear_pred)

# # Ridge Regression
# ridge_reg = Ridge(alpha=1.0)
# ridge_reg.fit(x_train, y_train)
# ridge_pred = ridge_reg.predict(x_test)
# ridge_score = r2_score(y_test, ridge_pred)

# # Lasso Regression
# lasso_reg = Lasso(alpha=1.0)
# lasso_reg.fit(x_train, y_train)
# lasso_pred = lasso_reg.predict(x_test)
# lasso_score = r2_score(y_test, lasso_pred)

# # ElasticNet Regression
# elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
# elastic_net_reg.fit(x_train, y_train)
# elastic_net_pred = elastic_net_reg.predict(x_test)
# elastic_net_score = r2_score(y_test, elastic_net_pred)

# # Decision Tree Regression
# dt_reg = DecisionTreeRegressor(random_state=0)
# dt_reg.fit(x_train, y_train)
# dt_pred = dt_reg.predict(x_test)
# dt_score = r2_score(y_test, dt_pred)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)
# rf_score = r2_score(y_test, rf_pred)

# # Gradient Boosting Regression
# gb_reg = GradientBoostingRegressor(random_state=0)
# gb_reg.fit(x_train, y_train)
# gb_pred = gb_reg.predict(x_test)
# gb_score = r2_score(y_test, gb_pred)

# # Support Vector Regression
# svr_reg = SVR(kernel='rbf')
# svr_reg.fit(x_train, y_train)
# svr_pred = svr_reg.predict(x_test)
# svr_score = r2_score(y_test, svr_pred)

# # Print the accuracy scores
# print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
# print("Ridge Regression Accuracy: {:.2f}%".format(ridge_score * 100))
# print("Lasso Regression Accuracy: {:.2f}%".format(lasso_score * 100))
# print("ElasticNet Regression Accuracy: {:.2f}%".format(elastic_net_score * 100))
# print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
# print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))
# print("Gradient Boosting Regression Accuracy: {:.2f}%".format(gb_score * 100))
# print("Support Vector Regression Accuracy: {:.2f}%".format(svr_score * 100))


# #Plot the simplified graph comparing the predicted and actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(x_test, y_test, color='b', label='Actual')
# plt.plot(x_test, linear_pred, color='r', label='Linear Regression')
# plt.plot(x_test, ridge_pred, color='g', label='Ridge Regression')
# plt.plot(x_test, lasso_pred, color='m', label='Lasso Regression')
# plt.plot(x_test, elastic_net_pred, color='c', label='ElasticNet Regression')
# plt.plot(x_test, dt_pred, color='y', label='Decision Tree Regression')
# plt.plot(x_test, rf_pred, color='k', label='Random Forest Regression')
# plt.plot(x_test, gb_pred, color='purple', label='Gradient Boosting Regression')
# plt.plot(x_test, svr_pred, color='orange', label='Support Vector Regression')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.idxmax()

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)

# # Get feature importances
# feature_importances = rf_reg.feature_importances_

# # Get the names of the features
# feature_names = ['Week_Num']

# # Plot feature importances
# plt.figure(figsize=(8, 6))
# plt.bar(feature_names, feature_importances)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importance Analysis')
# plt.show()


# import openpyxl
# from datetime import datetime, timedelta

# # Start and end dates
# start_date = datetime(2019, 7, 13)
# end_date = datetime(2023, 9, 30)

# # Calculate the number of weeks
# num_weeks = (end_date - start_date).days // 7

# # Create a new workbook and select the active sheet
# workbook = openpyxl.Workbook()
# sheet = workbook.active

# # Set the initial column value
# current_col = 1

# # Write headers
# sheet.cell(row=1, column=current_col, value="Date")

# # Write dates at weekly intervals
# for week in range(num_weeks + 1):
#     current_date = start_date + timedelta(weeks=week)
#     current_col += 1
#     sheet.cell(row=1, column=current_col, value=current_date)

# # Save the workbook to a file
# workbook.save("weekly_dates.xlsx")


# import openpyxl
# from datetime import datetime, timedelta

# # Start and end dates
# start_date = datetime(2019, 7, 13)
# end_date = datetime(2023, 9, 30)

# # Calculate the number of weeks
# num_weeks = (end_date - start_date).days // 7

# # Create a new workbook and select the active sheet
# workbook = openpyxl.Workbook()
# sheet = workbook.active

# # Set the initial row value
# current_row = 1

# # Write headers
# sheet.cell(row=current_row, column=1, value="Date")

# # Write dates at weekly intervals
# for week in range(num_weeks + 1):
#     current_date = start_date + timedelta(weeks=week)
#     current_row += 1
#     sheet.cell(row=current_row, column=1, value=current_date)

# # Save the workbook to a file
# workbook.save("weekly_dates_rows.xlsx")
# import csv
# from datetime import datetime, timedelta

# # Start and end dates
# start_date = datetime(2019, 7, 13)
# end_date = datetime(2023, 9, 30)

# # Calculate the number of weeks
# num_weeks = (end_date - start_date).days // 7

# # Create a list to store the dates
# dates = ["Date"]

# # Generate dates at weekly intervals
# for week in range(num_weeks + 1):
#     current_date = start_date + timedelta(weeks=week)
#     dates.append(current_date.strftime("%d-%m-%Y"))

# # Write the dates to a CSV file
# with open("weekly_dates.csv", mode="w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(dates)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train.ravel())

# # Obtain new data for predictions
# new_dates_data = pd.read_excel("C:/Users/swaya/OneDrive/Desktop/PROBABLITY AND MUSIC/weekly_dates_rows.xlsx")

# # Convert dates to datetime format
# new_dates_data['Date'] = pd.to_datetime(new_dates_data['Date'])

# # Convert dates to numerical representation
# ref_date = x_train.min().item()
# new_week_nums = (new_dates_data['Date'] - pd.to_datetime(ref_date)).dt.days

# # Convert the features of the new data to the same format as the training data
# new_features = new_week_nums.values.reshape(-1, 1)

# # Make predictions on the new data
# predictions = rf_reg.predict(new_features)

# # Add the predictions to the new dates data
# new_dates_data['Predictions'] = predictions

# # Print the new dates data with predictions
# print(new_dates_data)

# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], label='Actual Data')
# plt.plot(new_dates_data['Date'], new_dates_data['Predictions'], label='Predicted Data')
# plt.xlabel('Week')
# plt.ylabel('Count')
# plt.title('Weekly Count of Top Genre')
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)  # Adjust hyperparameters as needed
# rf_reg.fit(x_train, y_train.ravel())

# # Obtain new data for predictions
# new_dates_data = pd.read_excel("C:/Users/swaya/OneDrive/Desktop/PROBABLITY AND MUSIC/weekly_dates_rows.xlsx")
# new_dates_data['Week_Num'] = (new_dates_data['Date'] - ref_date).dt.days

# # Predict on new data
# new_dates_data['Predicted_Count'] = rf_reg.predict(new_dates_data['Week_Num'].values.reshape(-1, 1))

# # Plot the graph
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], label='Actual Data')
# plt.plot(new_dates_data['Date'], new_dates_data['Predicted_Count'], label='Predicted Data')
# plt.xlabel('Week')
# plt.ylabel('Count')
# plt.title('Weekly Count of Top Genre')
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
# rf_reg.fit(x_train, y_train.ravel())

# # Obtain new data for predictions
# new_dates_data = pd.read_excel("C:/Users/swaya/OneDrive/Desktop/PROBABLITY AND MUSIC/weekly_dates_rows.xlsx")
# new_dates_data['Week_Num'] = (new_dates_data['Date'] - ref_date).dt.days

# # Reshape the new data
# new_dates_data = new_dates_data['Week_Num'].values.reshape(-1, 1)

# # Predict on new data
# new_dates_data['Predicted_Count'] = rf_reg.predict(new_dates_data)

# # Plot the graph
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], label='Actual Data')
# plt.plot(new_dates_data['Date'], new_dates_data['Predicted_Count'], label='Predicted Data')
# plt.xlabel('Week')
# plt.ylabel('Count')
# plt.title('Weekly Count of Top Genre')
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[0]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
# rf_reg.fit(x_train, y_train.ravel())

# # Obtain new data for predictions
# new_dates_data = pd.read_excel("C:/Users/swaya/OneDrive/Desktop/PROBABLITY AND MUSIC/weekly_dates_rows.xlsx")
# new_dates_data['Week_Num'] = (new_dates_data['Date'] - ref_date).dt.days

# # Reshape the new data
# new_dates_data = new_dates_data['Week_Num'].values.reshape(-1, 1)

# # Predict on new data
# predictions = rf_reg.predict(new_dates_data)

# # Add the predictions to the new_dates_data DataFrame
# new_dates_data['Predicted_Count'] = predictions

# # Plot the graph
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], label='Actual Data')
# plt.plot(new_dates_data['Date'], new_dates_data['Predicted_Count'], label='Predicted Data')
# plt.xlabel('Week')
# plt.ylabel('Count')
# plt.title('Weekly Count of Top Genre')
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[0]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
# rf_reg.fit(x_train, y_train.ravel())

# # Obtain new data for predictions
# new_dates_data = pd.read_excel("C:/Users/swaya/OneDrive/Desktop/PROBABLITY AND MUSIC/weekly_dates_rows.xlsx")
# new_dates_data['Week_Num'] = (new_dates_data['Date'] - ref_date).dt.days

# # Reshape the new data
# new_dates_data = new_dates_data[['Week_Num']]

# # Predict on new data
# predictions = rf_reg.predict(new_dates_data)

# # Add the predictions to the new_dates_data DataFrame
# new_dates_data['Predicted_Count'] = predictions

# # Convert numerical representation back to dates
# new_dates_data['Date'] = ref_date + pd.to_timedelta(new_dates_data['Week_Num'], unit='D')

# # Plot the graph
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], label='Actual Data')
# plt.plot(new_dates_data['Date'], new_dates_data['Predicted_Count'], label='Predicted Data')
# plt.xlabel('Week')
# plt.ylabel('Count')
# plt.title('Weekly Count of Top Genre')
# plt.legend()
# plt.xticks(rotation=45)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Get the genre with the highest frequency
# top_genre = genre_counts.index[1]

# # Filter the dataset for the top genre
# top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# # Group and aggregate data at the weekly level for the top genre
# grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# # Convert dates to numerical representation
# ref_date = grouped['Week'].min()
# grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)


# # Linear Regression
# linear_reg = LinearRegression()
# linear_reg.fit(x_train, y_train)
# linear_pred = linear_reg.predict(x_test)
# linear_score = r2_score(y_test, linear_pred)

# # Ridge Regression
# ridge_reg = Ridge(alpha=1.0)
# ridge_reg.fit(x_train, y_train)
# ridge_pred = ridge_reg.predict(x_test)
# ridge_score = r2_score(y_test, ridge_pred)

# # Lasso Regression
# lasso_reg = Lasso(alpha=1.0)
# lasso_reg.fit(x_train, y_train)
# lasso_pred = lasso_reg.predict(x_test)
# lasso_score = r2_score(y_test, lasso_pred)

# # ElasticNet Regression
# elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
# elastic_net_reg.fit(x_train, y_train)
# elastic_net_pred = elastic_net_reg.predict(x_test)
# elastic_net_score = r2_score(y_test, elastic_net_pred)

# # Decision Tree Regression
# dt_reg = DecisionTreeRegressor(random_state=0)
# dt_reg.fit(x_train, y_train)
# dt_pred = dt_reg.predict(x_test)
# dt_score = r2_score(y_test, dt_pred)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)
# rf_score = r2_score(y_test, rf_pred)

# # Gradient Boosting Regression
# gb_reg = GradientBoostingRegressor(random_state=0)
# gb_reg.fit(x_train, y_train)
# gb_pred = gb_reg.predict(x_test)
# gb_score = r2_score(y_test, gb_pred)

# # Support Vector Regression
# svr_reg = SVR(kernel='rbf')
# svr_reg.fit(x_train, y_train)
# svr_pred = svr_reg.predict(x_test)
# svr_score = r2_score(y_test, svr_pred)


# # Print the accuracy scores
# print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
# print("Ridge Regression Accuracy: {:.2f}%".format(ridge_score * 100))
# print("Lasso Regression Accuracy: {:.2f}%".format(lasso_score * 100))
# print("ElasticNet Regression Accuracy: {:.2f}%".format(elastic_net_score * 100))
# print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
# print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))
# print("Gradient Boosting Regression Accuracy: {:.2f}%".format(gb_score * 100))
# print("Support Vector Regression Accuracy: {:.2f}%".format(svr_score * 100))


# df['Predicted']=rf_pred
# print(df)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import r2_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')

# # Group and aggregate data at the weekly level for the Artistss
# grouped = df.groupby(['Week', 'Artists']).size().reset_index(name='Count')

# # Get the Artists with the highest count
# top_Artists = grouped.groupby('Artists')['Count'].sum().idxmax()

# # Filter the dataset for the top Artists
# top_Artists_data = grouped[grouped['Artists'] == top_Artists]

# # Convert dates to numerical representation
# ref_date = top_Artists_data['Week'].min()
# top_Artists_data['Week_Num'] = (top_Artists_data['Week'] - ref_date).dt.days

# # Split the data into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(top_Artists_data['Week_Num'], top_Artists_data['Count'], test_size=0.2, random_state=0)

# # Reshape the training and test data
# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)

# # Linear Regression
# linear_reg = LinearRegression()
# linear_reg.fit(x_train, y_train)
# linear_pred = linear_reg.predict(x_test)
# linear_score = r2_score(y_test, linear_pred)

# # Ridge Regression
# ridge_reg = Ridge(alpha=1.0)
# ridge_reg.fit(x_train, y_train)
# ridge_pred = ridge_reg.predict(x_test)
# ridge_score = r2_score(y_test, ridge_pred)

# # Lasso Regression
# lasso_reg = Lasso(alpha=1.0)
# lasso_reg.fit(x_train, y_train)
# lasso_pred = lasso_reg.predict(x_test)
# lasso_score = r2_score(y_test, lasso_pred)

# # ElasticNet Regression
# elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
# elastic_net_reg.fit(x_train, y_train)
# elastic_net_pred = elastic_net_reg.predict(x_test)
# elastic_net_score = r2_score(y_test, elastic_net_pred)

# # Decision Tree Regression
# dt_reg = DecisionTreeRegressor(random_state=0)
# dt_reg.fit(x_train, y_train)
# dt_pred = dt_reg.predict(x_test)
# dt_score = r2_score(y_test, dt_pred)

# # Random Forest Regression
# rf_reg = RandomForestRegressor(random_state=0)
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)
# rf_score = r2_score(y_test, rf_pred)

# # Gradient Boosting Regression
# gb_reg = GradientBoostingRegressor(random_state=0)
# gb_reg.fit(x_train, y_train)
# gb_pred = gb_reg.predict(x_test)
# gb_score = r2_score(y_test, gb_pred)

# # Support Vector Regression
# svr_reg = SVR(kernel='rbf')
# svr_reg.fit(x_train, y_train)
# svr_pred = svr_reg.predict(x_test)
# svr_score = r2_score(y_test, svr_pred)


# # Print the accuracy scores
# print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
# print("Ridge Regression Accuracy: {:.2f}%".format(ridge_score * 100))
# print("Lasso Regression Accuracy: {:.2f}%".format(lasso_score * 100))
# print("ElasticNet Regression Accuracy: {:.2f}%".format(elastic_net_score * 100))
# print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
# print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))
# print("Gradient Boosting Regression Accuracy: {:.2f}%".format(gb_score * 100))
# print("Support Vector Regression Accuracy: {:.2f}%".format(svr_score * 100))

# # # Plot the simplified graph comparing the predicted and actual values
# # plt.figure(figsize=(12, 6))
# # plt.plot(top_Artists_data['Week'], top_Artists_data['Count'], color='blue', label='Actual')
# # #plt.plot(top_Artists_data['Week'], linear_reg.predict(top_Artists_data['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Linear Regression')
# # #plt.plot(top_Artists_data['Week'], dt_reg.predict(top_Artists_data['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Decision Tree Regression')
# # plt.plot(top_Artists_data['Week'], rf_reg.predict(top_Artists_data['Week_Num'].values.reshape(-1, 1)), color='orange', linewidth=2, label='Random Forest Regression')
# # plt.xlabel('Week')
# # plt.ylabel('Artists Count')
# # plt.title('Artists Count Over the Weeks - Top Artists: ' + top_Artists)
# # plt.legend()
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df=pd.read_csv("charts.csv")

# # Convert year column to datetime format
# df['Year'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# artist_counts = df['Artists'].value_counts()

# # Get the artist with the highest frequency
# top_artist = artist_counts.idxmax()

# # Filter the dataset for the top artist
# top_artist_data = df[df['Artists'] == top_artist]

# # Group and aggregate data at the yearly level for the top artist
# grouped = top_artist_data.groupby('Year').size().reset_index(name='Count')

# # Plot the graph for the top artist
# plt.figure(figsize=(10, 6))
# plt.plot(grouped['Year'], grouped['Count'], label=top_artist)

# plt.xlabel('Year')
# plt.ylabel('Artist Count')
# plt.title('Artist Count Over the Years - Top Artist: ' + top_artist)
# plt.legend()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')
# df = df.explode('Genre')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()

# # Select top 10 genres and group the rest as "Others"
# top_genres = genre_counts.head(6)
# other_count = genre_counts[6:].sum()
# top_genres['Others'] = other_count


# # Plot the genre distribution in a pie chart
# plt.figure(figsize=(12, 6))
# plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=90)
# plt.axis('equal')
# plt.title('Genre Distribution')
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
# df['Genres'] = df['Genre'].str.split(',')
# df = df.explode('Genres')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# print(genres_list)

# # Calculate the frequency of each genre
# genre_counts = pd.Series(genres_list).value_counts()


# # Select top 10 genres and group the rest as "Others"
# top_genres = genre_counts.head(15)
# other_count = genre_counts[15:].sum()
# top_genres['Others'] = other_count

# # print(top_genres)

# # # Plot the genre distribution in a pie chart
# # plt.figure(figsize=(12, 6))
# # plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=90)
# # plt.axis('equal')
# # plt.title('Top 10 Genre Distribution')
# # plt.show()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load and preprocess the dataset
# df = pd.read_csv("billboardHot100_1999-2019.csv")
# df['Genres'] = df['Genre'].str.split(',')

# # Flatten the list of genres
# genres_list = [genre for genres in df['Genres'] for genre in genres]

# # Create the feature matrix and target variable
# corpus = df['Artists'].fillna('') # Assuming 'Lyrics' column contains the text data
# target = df['Genre']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(corpus, target, test_size=0.2, random_state=42)

# # Vectorize the text data
# vectorizer = CountVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # Train a logistic regression model
# model = LogisticRegression()
# model.fit(X_train_vec, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test_vec)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")