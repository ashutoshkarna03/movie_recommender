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

# Now let's create a matrix that has the user ids on one access and the movie title on another axis
movie_mat = user_data.pivot_table(index='user_id', columns='title', values='rating')
pprint(movie_mat.head())

# let's try recommendation for movie `Star Wars (1977)`

# let's grab the user's rating for this movie
starwars_user_ratings = movie_mat['Star Wars (1977)']
pprint(starwars_user_ratings.head(10))
# get the movies similar to starwars using co-relation
similar_to_starwars = movie_mat.corrwith(starwars_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
pprint(corr_starwars.head())

# let's check the result by sorting Correlation
pprint(corr_starwars.sort_values('Correlation', ascending=False).head(10))

# we find lots of movie having Correlation 1, but the number of ratings would be less than 100, so we need to filter out those also
corr_starwars = corr_starwars.join(ratings['no_of_ratings'])
pprint(corr_starwars.head())

corr_starwars = corr_starwars[corr_starwars['no_of_ratings'] > 100]
pprint(corr_starwars.sort_values('Correlation', ascending=False).head())
recommendation_for_starwars = corr_starwars.sort_values('Correlation', ascending=False).head().index
print('##############################')
pprint(recommendation_for_starwars)
print('##############################')

# similarly recommendation for other movies can also be made
