# Library

# Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import geopandas as gpd
import pycountry
import re
from collections import Counter
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn import preprocessing
from collections import defaultdict

from math import pi
import os

# Import Dataset

Player_Data = pd.read_csv('player_Data.csv')
print(Player_Data.shape)
print(Player_Data.isnull().sum().any())
Player_Data = Player_Data.dropna()
Player_Data = Player_Data.sort_values(by='Potential', ascending=False).reset_index()
print(Player_Data.shape)

Player_Data.head()

# Exploratory Data Analysis


# HeatMap
player_characteristics = ['Age', 'Overall', 'Potential',
                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 
                'Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 
                'HeadingAccuracy', 'Interceptions','International.Reputation',
                'Jumping', 'LongPassing', 'LongShots',
                'Marking', 'Penalties', 'Position', 'Positioning',
                'ShortPassing', 'ShotPower', 'Skill.Moves', 'SlidingTackle',
                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',
                'Volleys']



plt.figure(figsize= (30, 16))


hm=sns.heatmap(Player_Data[player_characteristics].corr(), annot = True, linewidths=.5, cmap='Greens')
hm.set_title(label='Heatmap of dataset', fontsize=20)
hm;

# Correlation between Acceleration and other
def make_scatter(df):
    feats = ('Agility', 'Balance', 'Dribbling','SprintSpeed')
    
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/4+1, 4, index+1)
        ax = sns.regplot(x='Acceleration', y=feat, data=df)

plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=0.4)

make_scatter(Player_Data)

# All of position
a4_dims = (15, 8.7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(x = 'Position', data = Player_Data, palette = 'hls');
ax.set_title(label='Count of players on the position', fontsize=10);

AM – attacking midfielder.

SW – sweeper. A player with both defensive and offensive tasks. He is given a free role and can serve in some degree as a playmaker and should also fall back behind the defensive line when the opposite side attack.

CB – center back. Normally one or two center backs are used in a formation.

CF – center forward. The attacker that is positioned in the middle of the offensive line. In modern football it has become common to only use one or two attackers; therefore a center forward may not be quite relevant as a description.

LB – left back. Is positioned on the left part of the defensive line.

RB – right back. Is positioned on the right part of the defensive line.

FB – fullback. Another name for the defensive player that either plays on the left side (left back) or the right side (right back).

LWB – left wing back. Positioned in front of the left back and out on the “wing”.

RWB – right wing back. Positioned in front of the right back.

D – defender.

DM – defensive midfielder.

CM – center midfielder.

F – forward.

GK – goalkeeper. Often only G is used.

LW – left wing, similar to the left wing back, but usually with a primarily offensive task. In other word an offensive wing midfielder.

RW – right wing. The same as the left wing, but on the opposite wing.

M – midfielder.

WF – wing forward. An attacker in offensive position on the wing. As with the center forward, the wing forward has been less common in the modern game, but could be present in a 4-3-3 formation.

ST – Striker. A similar function as the center and wing forward.

IF – Inside forward. In the old days an offensive line could consist of five attackers and include two inside forwards that were positioned between the wing forwards and the center forward and normally a little behind the other three.

OL – Outside left. Same as left-winger.

OR – Outside right. Same as left-winger.

# Feature Engineering

Player_Data = Player_Data.reset_index()
del Player_Data['index']
hasil =[]
text1 = Player_Data['Position']
for i in text1:
    if i=='CB' or i=='RCB' or i=='LCB': #Centre Beck
        hasil.append('CB')
    elif i=='LB' or i=='RB' or i=='RWB' or i=='LWB':  #Side Back
        hasil.append('SB')
    elif i=='CDM' or i=='LDM' or i=='RDM' or i=='CM' or i=='RCM' or i=='LCM': #Defensive Midfielder
        hasil.append('DM')
    elif i=='RAM' or i=='CAM' or i=='LAM': #Attacking Midfielder 
        hasil.append('AM')
    else:   #Goal Machines
        hasil.append('GM')
series = pd.DataFrame(hasil)
Player_Data['GroupPosition'] = series

# All of position
a4_dims = (15, 8.7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(x = 'GroupPosition', data = Player_Data, palette = 'hls');
ax.set_title(label='Count of players on the position', fontsize=10);

counter_CB = Player_Data[Player_Data['GroupPosition'] == 'CB']
counter_SB = Player_Data[Player_Data['GroupPosition'] == 'SB']
counter_DM = Player_Data[Player_Data['GroupPosition'] == 'DM']
counter_AM = Player_Data[Player_Data['GroupPosition'] == 'AM']
counter_GM = Player_Data[Player_Data['GroupPosition'] == 'GM']

player_features = [
                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 
                'Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 
                'HeadingAccuracy', 'Interceptions','International.Reputation',
                'Jumping', 'LongPassing', 'LongShots',
                'Marking', 'Penalties', 'Position', 'Positioning',
                'ShortPassing', 'ShotPower', 'Skill.Moves', 'SlidingTackle',
                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',
                'Volleys'
]

lis = []

for i, val in Player_Data.groupby(Player_Data['GroupPosition'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {},{}'.format(i, *tuple(val.nlargest(4).index)))
    lis.append(val.nlargest(4))

cb = ['StandingTackle', 'Marking', 'Strength','Interceptions']
sb = ['SprintSpeed', 'Stamina', 'Acceleration','StandingTackle']
dm = ['ShortPassing', 'BallControl','Stamina','LongPassing']
am = ['BallControl','Agility','Dribbling','Vision']
gm = ['Acceleration', 'Finishing','Agility','Dribbling']

# Defence Player Recommendation

CB_Data = pd.concat([counter_CB[['Name','Age','Potential','Club']],counter_CB[cb],counter_CB['Release.Clause']],1).reset_index()
CB_Data2 = CB_Data.copy()

df = counter_CB[['Potential','StandingTackle','Marking','Strength','Interceptions']]
    
def clustering(data):
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)
    range_n_clusters = list (range(2,5))
    for n_clusters in range_n_clusters:
        clusterer = KMeans (n_clusters=n_clusters)
        preds = clusterer.fit_predict(data_transformed)
        centers = clusterer.cluster_centers_

        score = silhouette_score (data_transformed, preds, metric='euclidean')
        print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
dfData = clustering(df)
kmeans1 = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=123)
predDef_y = kmeans1.fit_predict(df)
yDef = pd.DataFrame(predDef_y, columns=['Class']).reset_index(drop=True)
CB_Data = pd.concat([CB_Data,yDef],1)
CB_Data.head(5)

dfCB = CB_Data[['StandingTackle','Marking','Strength','Interceptions']]
target = CB_Data['Class']

# Predict

X_train, X_test, y_train, y_test = train_test_split(dfCB, target, test_size=0.3, random_state=123)
param_rnd={
        'n_estimators':randint(low=1,high=5),
        'max_features':randint(low=1,high=2),
    }
forest_reg=RandomForestClassifier(random_state=123)

rnd_def=RandomizedSearchCV(forest_reg,param_distributions=param_rnd,
                                  n_iter=10,cv=10,scoring='neg_mean_squared_error',random_state=123)
rnd_def.fit(X_train,y_train)
final_model_rnd=rnd_def.best_estimator_
final_predictions=final_model_rnd.predict(X_test)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_rmse)
print(metrics.r2_score(y_test, final_predictions))
accuracy_score(y_test,final_predictions)

def train_val(df, target, num_split):
    global final_model_rnd
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=num_split, random_state=123)
    param_rnd={
            'n_estimators':randint(low=1,high=5),
            'max_features':randint(low=1,high=2),
        }
    forest_reg=RandomForestClassifier(random_state=123)

    rnd_def=RandomizedSearchCV(forest_reg,param_distributions=param_rnd,
                                      n_iter=10,cv=10,scoring='neg_mean_squared_error',random_state=123)
    rnd_def.fit(X_train,y_train)
    final_model_rnd=rnd_def.best_estimator_
    final_predictions=final_model_rnd.predict(X_test)
    print(metrics.r2_score(y_test, final_predictions))
    
def test():
    result = int(final_model_rnd.predict([[65,66,79,65]]))
    print(result)

#How old is he?
#How about the potential?
def analyze(df, age, potential, result):
    Age = df[(df['Age'] >= age - 1) & (df['Age'] <= age + 1)]
    ptnial = Age[(Age['Potential'] >= potential - 5) & (Age['Potential'] <= potential + 5) ]
    ptnial = ptnial[ptnial['Class'] == result]
    return ptnial

# Demo

CB_Data[CB_Data['Name'] == 'Sergio Ramos']

result = int(final_model_rnd.predict([[90,88,88,88]]))
analyze(CB_Data,27,89,result).head()
