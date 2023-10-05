import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Load and preprocess the dataset
df = pd.read_csv("billboardHot100_1999-2019.csv")
df['Week'] = pd.to_datetime(df['Week'], format='%d-%m-%Y')
df['Genres'] = df['Genre'].str.split(',')

# Flatten the list of genres
genres_list = [genre for genres in df['Genres'] for genre in genres]

# Calculate the frequency of each genre
genre_counts = pd.Series(genres_list).value_counts()

# Get the genre with the highest frequency
top_genre = genre_counts.index[0]

# Filter the dataset for the top genre
top_genre_data = df[df['Genres'].apply(lambda x: top_genre in x)]

# Group and aggregate data at the weekly level for the top genre
grouped = top_genre_data.groupby('Week').size().reset_index(name='Count')

# Convert dates to numerical representation
ref_date = grouped['Week'].min()
grouped['Week_Num'] = (grouped['Week'] - ref_date).dt.days

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(grouped['Week_Num'], grouped['Count'], test_size=0.2, random_state=0)

# Reshape the training and test data
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
linear_pred = linear_reg.predict(x_test)
linear_score = r2_score(y_test, linear_pred)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(x_train, y_train)
ridge_pred = ridge_reg.predict(x_test)
ridge_score = r2_score(y_test, ridge_pred)

# Lasso Regression
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(x_train, y_train)
lasso_pred = lasso_reg.predict(x_test)
lasso_score = r2_score(y_test, lasso_pred)

# ElasticNet Regression
elastic_net_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net_reg.fit(x_train, y_train)
elastic_net_pred = elastic_net_reg.predict(x_test)
elastic_net_score = r2_score(y_test, elastic_net_pred)

# Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_train, y_train)
dt_pred = dt_reg.predict(x_test)
dt_score = r2_score(y_test, dt_pred)

# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(x_train, y_train)
rf_pred = rf_reg.predict(x_test)
rf_score = r2_score(y_test, rf_pred)

# Gradient Boosting Regression
gb_reg = GradientBoostingRegressor(random_state=0)
gb_reg.fit(x_train, y_train)
gb_pred = gb_reg.predict(x_test)
gb_score = r2_score(y_test, gb_pred)

# Support Vector Regression
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train, y_train)
svr_pred = svr_reg.predict(x_test)
svr_score = r2_score(y_test, svr_pred)


# Print the accuracy scores
print("Linear Regression Accuracy: {:.2f}%".format(linear_score * 100))
print("Ridge Regression Accuracy: {:.2f}%".format(ridge_score * 100))
print("Lasso Regression Accuracy: {:.2f}%".format(lasso_score * 100))
print("ElasticNet Regression Accuracy: {:.2f}%".format(elastic_net_score * 100))
print("Decision Tree Regression Accuracy: {:.2f}%".format(dt_score * 100))
print("Random Forest Regression Accuracy: {:.2f}%".format(rf_score * 100))
print("Gradient Boosting Regression Accuracy: {:.2f}%".format(gb_score * 100))
print("Support Vector Regression Accuracy: {:.2f}%".format(svr_score * 100))


# # Plot the simplified graph comparing the predicted and actual values
# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], color='blue', label='Actual')
# plt.plot(grouped['Week'], linear_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Linear Regression')
# #plt.plot(grouped['Week'], dt_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Decision Tree Regression')
# #plt.plot(grouped['Week'], rf_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='orange', linewidth=2, label='Random Forest Regression')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()


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




# plt.figure(figsize=(12, 6))
# plt.plot(grouped['Week'], grouped['Count'], color='blue', label='Actual')
# #plt.plot(grouped['Week'], linear_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Linear Regression')
# #plt.plot(grouped['Week'], dt_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='red', linewidth=2, label='Decision Tree Regression')
# plt.plot(grouped['Week'], rf_reg.predict(grouped['Week_Num'].values.reshape(-1, 1)), color='orange', linewidth=2, label='Random Forest Regression')
# plt.xlabel('Week')
# plt.ylabel('Genre Count')
# plt.title('Genre Count Over the Weeks - Top Genre: ' + top_genre)
# plt.legend()
# plt.show()