#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sqlalchemy import null

#import data, data was seperated by tab instead of comma
artistsDf = pd.read_csv('musicoset_data\musicoset_metadata\musicoset_metadata/artists.csv', sep='\t')
songsDf = pd.read_csv('musicoset_data\musicoset_metadata\musicoset_metadata\songs.csv', sep='\t')
songFeaturesDf = pd.read_csv('musicoset_data\musicoset_songfeatures\musicoset_songfeatures/acoustic_features.csv', sep='\t')

#preprocessing: merge datasets
SongsWFeautresDf = pd.merge(left=songsDf, right=songFeaturesDf, how='inner', on='song_id')
regextoomit = '|'.join(['{', '}', "'"])
artistsID = SongsWFeautresDf['artists'].str.replace(regextoomit,'').str.split(':', expand=True)
artistsID.drop(artistsID.columns[2:], axis=1, inplace=True)
artistsID.columns = ['artist_id', 'artist_name']

allSongsDf = pd.concat([SongsWFeautresDf, artistsID], axis=1)
allSongsDf.drop(['billboard', 'artists'],axis=1, inplace=True)

artistsDf.drop(['followers', 'popularity', 'artist_type','image_url'], axis=1, inplace=True)

sgaDf = pd.merge(left=allSongsDf, right=artistsDf, how ='inner', on='artist_id')

#drop rows without genre info
sgaDf['main_genre'].replace('-', np.NaN, inplace=True)
sgaDf['main_genre'].isnull().value_counts()
sgaDf = sgaDf[sgaDf['main_genre'].notna()]

#Create a dummy variable for metal genres
Metal = []

for genre in sgaDf['main_genre']:
    genre = genre.lower()
    if 'metal' in genre:
        Metal.append(1)
    else:
        Metal.append(0)
    
sgaDf['Metal'] = Metal

sgaDf[sgaDf['Metal'] == 1]

#Define predictors and dependent variable
sgaDf.columns
independentVars = ['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
dependentVar = ['Metal']

#Let's see if song features can predict genre
sns.pairplot(data=sgaDf, x_vars=independentVars, y_vars=dependentVar)
plt.show()

#Split the data for training and testing
X = sgaDf[['duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']]
y = sgaDf['Metal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

#Time for regression analysis
logm = LogisticRegression()
logm.fit(X_train, y_train)

predictions = logm.predict(X_test)

print(classification_report(y_test, predictions))
