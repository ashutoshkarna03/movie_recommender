import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns
sns.set_style('white')


# load user data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
user_data = pd.read_csv('data/u.data', sep='\t', names=column_names)

# check the head of df
print('user data frame: ')
pprint(user_data.head())

# read the movie title data and attach this to user data frame in separate column
movie_titles = pd.read_csv('data/Movie_Id_Titles')
print('movie title data frame: ')
pprint(movie_titles.head())

user_data = pd.merge(user_data, movie_titles, on='item_id')
print('user data after adding movie title: ')
pprint(user_data.head())

# lets create another data frame which contains number of ratings and average ratings of each movie
# add mean rating of each movie
ratings = pd.DataFrame(user_data.groupby('title')['rating'].mean())
pprint(ratings.head())
# also add number of ratings
ratings['no_of_ratings'] = pd.DataFrame(user_data.groupby('title')['rating'].count())
pprint(ratings.head())

# visualize movie data
# histogram of no_of_ratings
plt.figure(figsize=(10, 4))
ratings['no_of_ratings'].hist(bins=70)
plt.show()

# histogram of average ratings
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)
plt.show()

# joint plot of average ratings vs no of ratings
sns.jointplot(x='rating', y='no_of_ratings', data=ratings, alpha=0.5)
plt.show()
