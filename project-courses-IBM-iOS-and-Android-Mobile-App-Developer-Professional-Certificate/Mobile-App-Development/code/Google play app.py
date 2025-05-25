
# coding: utf-8

# # Google Play App Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from sklearn.preprocessing import LabelEncoder


# In[2]:


Gapp_info = pd.read_csv('googleplaystore.csv')
Gapp_review = pd.read_csv('googleplaystore_user_reviews.csv')


# In[3]:


Gapp_info.head()


# ## Google Play App Informations

# In[4]:


# rename 
Gapp_info.rename(columns = {'Type': 'Price_Type','Content Rating': 'Content_Rating', 'Last Updated':'Last_Updated', 'Current Ver': 'Current_Ver', 'Android Ver':'Android_Ver'}, inplace= True)
Gapp_info.head()


# In[5]:


# check null value
Gapp_info.isnull().sum()


# In[6]:


# remove rating = null
Gapp_info.dropna(subset=['Rating'], inplace = True)


# In[7]:


print(Gapp_info.duplicated().sum())
Gapp_info.drop_duplicates(keep = 'first', inplace = True)


# In[8]:


Gapp_info.duplicated().sum()


# In[9]:


Gapp_info_re = Gapp_info.drop(columns = ['Genres', 'Last_Updated', 'Current_Ver','Android_Ver'])
Gapp_info_re.head()


# In[10]:


Gapp_info_re.duplicated().sum()


# In[10]:


Gapp_info_re.drop_duplicates(keep = 'first', inplace = True)


# In[11]:


plt.figure(figsize=(16,8))
Gapp_info.Rating.value_counts().sort_index().plot(kind = 'bar')
plt.title('App Rating')
plt.show()


# In[12]:


Gapp_info_re = Gapp_info_re[Gapp_info_re.Rating != 19.0]


# In[14]:


plt.figure(figsize=(6,8))
Gapp_info.Price_Type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Pay or not')
plt.show()


# In[15]:


plt.figure(figsize=(16,8))
Gapp_info.Price.value_counts().plot(kind = 'bar')
plt.title('App Price')
plt.show()


# In[13]:


Gapp_info_re = Gapp_info_re[Gapp_info_re.Price != 'Everyone']


# In[17]:


plt.figure(figsize=(20,8))
Gapp_info.Genres.value_counts().plot(kind = 'bar')
plt.title('App Genres')
plt.show()


# In[18]:


plt.figure(figsize=(20,8))
Gapp_info.Category.value_counts().plot(kind = 'bar')
plt.title('App Category')
plt.show()


# In[14]:


Gapp_info_re = Gapp_info_re[Gapp_info_re.Category != '1.9']


# In[15]:


Gapp_info_re = Gapp_info_re.replace('ART_AND_DESIGN', 1)
Gapp_info_re = Gapp_info_re.replace('AUTO_AND_VEHICLES', 2)
Gapp_info_re = Gapp_info_re.replace('BEAUTY', 3)
Gapp_info_re = Gapp_info_re.replace('BOOKS_AND_REFERENCE', 4)
Gapp_info_re = Gapp_info_re.replace('BUSINESS', 5)
Gapp_info_re = Gapp_info_re.replace('COMICS', 6)
Gapp_info_re = Gapp_info_re.replace('COMMUNICATION', 7)
Gapp_info_re = Gapp_info_re.replace('DATING', 8)
Gapp_info_re = Gapp_info_re.replace('EDUCATION', 9)
Gapp_info_re = Gapp_info_re.replace('ENTERTAINMENT', 10)
Gapp_info_re = Gapp_info_re.replace('EVENTS', 11)
Gapp_info_re = Gapp_info_re.replace('FINANCE', 12)
Gapp_info_re = Gapp_info_re.replace('FOOD_AND_DRINK', 13)
Gapp_info_re = Gapp_info_re.replace('HEALTH_AND_FITNESS', 14)
Gapp_info_re = Gapp_info_re.replace('HOUSE_AND_HOME', 15)
Gapp_info_re = Gapp_info_re.replace('LIBRARIES_AND_DEMO', 16)
Gapp_info_re = Gapp_info_re.replace('LIFESTYLE', 17)
Gapp_info_re = Gapp_info_re.replace('GAME', 18)
Gapp_info_re = Gapp_info_re.replace('FAMILY', 19)
Gapp_info_re = Gapp_info_re.replace('MEDICAL', 20)
Gapp_info_re = Gapp_info_re.replace('SOCIAL', 21)
Gapp_info_re = Gapp_info_re.replace('SHOPPING', 22)
Gapp_info_re = Gapp_info_re.replace('PHOTOGRAPHY', 23)
Gapp_info_re = Gapp_info_re.replace('SPORTS', 24)
Gapp_info_re = Gapp_info_re.replace('TRAVEL_AND_LOCAL', 25)
Gapp_info_re = Gapp_info_re.replace('TOOLS', 26)
Gapp_info_re = Gapp_info_re.replace('PERSONALIZATION', 27)
Gapp_info_re = Gapp_info_re.replace('PRODUCTIVITY', 28)
Gapp_info_re = Gapp_info_re.replace('PARENTING', 29)
Gapp_info_re = Gapp_info_re.replace('WEATHER', 30)
Gapp_info_re = Gapp_info_re.replace('VIDEO_PLAYERS', 31)
Gapp_info_re = Gapp_info_re.replace('NEWS_AND_MAGAZINES', 32)
Gapp_info_re = Gapp_info_re.replace('MAPS_AND_NAVIGATION', 33)


# In[22]:


plt.figure(figsize=(10,8))
Gapp_info.Content_Rating.value_counts().plot(kind = 'bar')
plt.title('Age group the App is targeted at')
plt.show()


# In[17]:


print(Gapp_info.Content_Rating.value_counts())
Gapp_info_re['Content_Rating'].unique()


# In[16]:


Gapp_info_re = Gapp_info_re[Gapp_info_re.Content_Rating != 'Unrated']
Gapp_info_re = Gapp_info_re.replace('Everyone', 1)
Gapp_info_re = Gapp_info_re.replace('Teen', 2)
Gapp_info_re = Gapp_info_re.replace('Everyone 10+', 3)
Gapp_info_re = Gapp_info_re.replace('Mature 17+', 4)
Gapp_info_re = Gapp_info_re.replace('Adults only 18+', 5)


# In[18]:


# Install Count
Gapp_info_re = Gapp_info_re[Gapp_info_re.Installs != 'Free']
Gapp_info_re['Installs'].unique()


# In[19]:


plt.figure(figsize=(10,8))
Gapp_info_re.Installs.value_counts().plot(kind = 'bar')
plt.title('Installation Count')
plt.show()


# In[20]:


Gapp_info_re = Gapp_info_re.replace('10,000+', 10000)
Gapp_info_re = Gapp_info_re.replace('500,000+', 500000)
Gapp_info_re = Gapp_info_re.replace('5,000,000+', 5000000)
Gapp_info_re = Gapp_info_re.replace('50,000,000+', 50000000)
Gapp_info_re = Gapp_info_re.replace('100,000+', 100000)
Gapp_info_re = Gapp_info_re.replace('1,000,000+', 1000000)
Gapp_info_re = Gapp_info_re.replace('10,000,000+', 10000000)
Gapp_info_re = Gapp_info_re.replace('5,000+', 5000)
Gapp_info_re = Gapp_info_re.replace('100,000,000+', 100000000)
Gapp_info_re = Gapp_info_re.replace('1,000,000,000+', 1000000000)
Gapp_info_re = Gapp_info_re.replace('1,000+', 1000)
Gapp_info_re = Gapp_info_re.replace('500,000,000+', 500000000)
Gapp_info_re = Gapp_info_re.replace('100+', 100)
Gapp_info_re = Gapp_info_re.replace('500+', 500)
Gapp_info_re = Gapp_info_re.replace('10+', 10)
Gapp_info_re = Gapp_info_re.replace('5+', 5)
Gapp_info_re = Gapp_info_re.replace('50+', 50)
Gapp_info_re = Gapp_info_re.replace('1+', 1)
Gapp_info_re = Gapp_info_re.replace('50,000+', 50000)


# In[21]:


install = []

for i in Gapp_info_re.Installs:
    if i < 1000: 
        install.append(500)
    elif 1000 <= i < 10000:
        install.append(5000)
    elif 10000 <= i < 100000:
        install.append(50000)
    elif 100000 <= i < 1000000:
        install.append(500000)
    elif 1000000 <= i < 10000000:
        install.append(5000000)
    elif 10000000 <= i < 100000000:
        install.append(50000000)
    elif 100000000 <= i < 1000000000:
        install.append(500000000)
    else:
        install.append(1000000000)
        
Gapp_info_re['Installs_re'] = install


# In[22]:


print(Gapp_info_re['Price_Type'].value_counts())
Gapp_info_re = Gapp_info_re.replace('Free', 0)
Gapp_info_re = Gapp_info_re.replace('Paid', 1)
Gapp_info_re['Price_Type'].unique()


# In[23]:


Gapp_info_re['Reviews'] = Gapp_info_re['Reviews'].apply(float)
Gapp_info_re['Installs'] = Gapp_info_re['Installs'].apply(int)

Gapp_info_re.info()


# In[24]:


Gapp_info_re.head()


# In[25]:


Gapp_info_re.drop_duplicates(subset = 'App', keep = 'first', inplace = True)


# In[26]:


Gapp_info_re.reset_index(inplace = True, drop = True)
Gapp_info_re.head()


# ### Chenge size into float

# In[27]:


S = []
for i in Gapp_info_re.Size:
    if i[-1] == 'M':
        s = i[:-1]
        s = float(s)*1000000
        S.append(s)
    elif i[-1] == 'k':
        s = i[:-1]
        s = float(s)*1000
        S.append(s)
    else:
        S.append(0)


# In[28]:


S = pd.DataFrame(S, columns = ['S'], index = Gapp_info_re.index)
M = S.mean()
print("%.f" % M)


# In[29]:


Size = []
for i in Gapp_info_re.Size:
    if i[-1] == 'M':
        s = i[:-1]
        s = float(s)*1000000
        Size.append(s)
    elif i == 'Varies with device':
        Size.append(float(18653076))
    else:
        s = i[:-1]
        s = float(s)*1000
        Size.append(s)


# In[30]:


Size = pd.DataFrame(Size, columns = ['Size'], index = Gapp_info_re.index)
Size.head()


# In[31]:


Gapp_info_re['Size_r'] = Size


# ### Separate $ and number in Price column

# In[32]:


Price = []

for i in Gapp_info_re.Price: 
    if i == '0':
        Price.append(0.0)
    else:
        p = i[1:]
        Price.append(float(p))


# In[33]:


Price = pd.DataFrame(Price, columns = ['Price_r'], index = Gapp_info_re.index)
Price.head()


# In[34]:


Gapp_info_re['Price_r'] = Price


# ### New column for Rating

# In[35]:


Rating_r = []

for i in range(Gapp_info_re.Rating.shape[0]):
    r = Gapp_info_re.Rating[i]
    if r < 2.0:
        Rating_r.append(float(1.5))
    elif 2.0 <= r < 3.0:
        Rating_r.append(float(2.5))
    elif 3.0 <= r < 4.0:
        Rating_r.append(float(3.5))
    elif 4.0 <= r <5.0:
        Rating_r.append(float(4.5))
    else:
        Rating_r.append(float(5.0))
    


# In[36]:


Rating_re = pd.DataFrame(Rating_r, columns = ['Rating_r'], index = Gapp_info_re.index)
Rating_re.head()


# In[37]:


Gapp_info_re['Rating_r'] = Rating_re


# In[38]:


plt.figure(figsize=(8,6))
Gapp_info_re.Rating_r.value_counts().sort_index().plot(kind = 'bar')
plt.title('App Rating')
plt.show()


# In[74]:


rating = []
for i in Gapp_info_re.Rating:
    if i >= 4.0:
        rating.append(1)
    else:
        rating.append(0)
        
Gapp_info_re['rating_type'] = rating
Gapp_info_re.head()


# ### Correlation Heatmap

# In[49]:


f, ax = plt.subplots(figsize=(12, 12))
corr = Gapp_info_re.corr()
sns.heatmap(corr,annot=True,linewidths=.5, fmt= '.2f',mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(400,500, as_cmap=True), square=True, ax=ax)
plt.show()


# In[50]:


# Pair Grid - showing the relationships in the dataset


pairplot = sns.PairGrid(Gapp_info_re, hue = 'Rating', palette = 'Reds')
pairplot = pairplot.map(plt.scatter)
#pairplot = pairplot.map_diag(plt.hist, edgecolor = 'w')
pairplot = pairplot.add_legend()


# ### Regression Model comprised of App Rating and other variables

# In[51]:


#Regression, Rating
cols = ['Category', 'Rating', 'Reviews', 'Installs', 'Price_Type', 'Content_Rating', 'Size_r', 'Price_r']

target_idx = len(cols) - 7
for exp_var_idx in range(len(cols)):
    
    sns.lmplot(x = Gapp_info_re[cols].columns[exp_var_idx], y = Gapp_info_re[cols].columns[target_idx], data = Gapp_info_re[cols], line_kws={"color":"red"})  
    
plt.tight_layout()
plt.show()


# In[52]:


#Regression, Installs
cols = ['Category', 'Rating', 'Reviews', 'Installs', 'Price_Type', 'Content_Rating', 'Size_r', 'Price_r']

target_idx = len(cols) - 5
for exp_var_idx in range(len(cols)):
    
    sns.lmplot(x = Gapp_info_re[cols].columns[exp_var_idx], y = Gapp_info_re[cols].columns[target_idx], data = Gapp_info_re[cols], line_kws={"color":"red"})  
    
plt.tight_layout()
plt.show()


# In[53]:


y = Gapp_info.iloc[:, 1:2].values.ravel()

class_le = LabelEncoder()
y = class_le.fit_transform(y)
Category = pd.DataFrame(y, columns = ['Category'])
Category.head()


# ### Comparison of Price Type(pay or not) and other Variables

# In[54]:


# Price vs. Category
pd.crosstab(Gapp_info.Category,Gapp_info_re.Price_Type).plot(kind='bar')
plt.title('Price vs. Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# In[55]:


# Price vs. Rating 
pd.crosstab(Gapp_info.Rating,Gapp_info_re.Price_Type).plot(kind='bar')
plt.title('Price vs. Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[56]:


# Price vs. Rating 
pd.crosstab(Gapp_info.Installs,Gapp_info_re.Price_Type).plot(kind='bar')
plt.title('Price vs. Install Count')
plt.xlabel('Install Count')
plt.ylabel('Count')
plt.show()


# ### Top Rating (Rating >= 4.0)

# In[75]:


Top_Rating = Gapp_info_re[Gapp_info_re['Rating'] >= 4.0]
Top_Rating.reset_index(inplace = True)
Top_Rating.head(10)


# In[58]:


plt.figure(figsize=(10,8))
Top_Rating.Installs.value_counts().plot(kind = 'bar')
plt.title('Install Count(rating >= 4.0)')
plt.show()


# In[59]:


plt.figure(figsize=(6,6))
Top_Rating.Price_Type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Price Type (Pay or not)')
plt.show()


# In[60]:


plt.figure(figsize=(16,8))
Top_Rating.Category.value_counts().plot(kind = 'bar')
plt.title('App Category')
plt.show()
'''('ART_AND_DESIGN', 1)
('AUTO_AND_VEHICLES', 2)
('BEAUTY', 3)
('BOOKS_AND_REFERENCE', 4)
('BUSINESS', 5)
('COMICS', 6)
('COMMUNICATION', 7)
('DATING', 8)
('EDUCATION', 9)
('ENTERTAINMENT', 10)
('EVENTS', 11)
('FINANCE', 12)
('FOOD_AND_DRINK', 13)
('HEALTH_AND_FITNESS', 14)
('HOUSE_AND_HOME', 15)
('LIBRARIES_AND_DEMO', 16)
('LIFESTYLE', 17)
('GAME', 18)
('FAMILY', 19)
('MEDICAL', 20)
('SOCIAL', 21)
('SHOPPING', 22)
('PHOTOGRAPHY', 23)
('SPORTS', 24)
('TRAVEL_AND_LOCAL', 25)
('TOOLS', 26)
('PERSONALIZATION', 27)
('PRODUCTIVITY', 28)
('PARENTING', 29)
('WEATHER', 30)
('VIDEO_PLAYERS', 31)
('NEWS_AND_MAGAZINES', 32)
('MAPS_AND_NAVIGATION', 33)

Top 5: Family, Games, Tools, Personalization, Medical
'''


# ### Installation Counts

# In[65]:


Install_count = Gapp_info_re.sort_values('Installs',ascending = False).head(500)
Install_count.reset_index(inplace = True)
Install_count.head(10)


# In[66]:


plt.figure(figsize=(6,6))
Install_count.Price_Type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Price Type (Pay or not)')
plt.show()


# In[67]:


plt.figure(figsize=(16,8))
Install_count.Category.value_counts().plot(kind = 'bar')
plt.title('App Category')
plt.show()

'''
Top 5:
Game, Family, Tools, Photography, Shopping
'''


# In[68]:


plt.figure(figsize=(8,8))
Install_count.Content_Rating.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Content Rating')
plt.show()


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn import neighbors


# # Installation Analysis

# In[220]:


plt.figure(figsize=(6,6))
Gapp_info_re.Installs_re.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Installation Count')
plt.show()


# ## Feature Importance for Install Count

# In[192]:


fm_installcount = ['Category', 'Price_Type', 'Price_r', 'Content_Rating', 'Size_r']
y = Gapp_info_re.iloc[:,9:10].values.reshape(-1, 1)
X_df = Gapp_info_re.ix[:, fm_installcount]
X = Gapp_info_re.ix[:, fm_installcount].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Random forest classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train) 

importances = rfc.feature_importances_

# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X_df.columns[:])

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

f_importances


# In[193]:


fscores = []
for k in range(1, 6):
    X = X_df[f_importances.index[:k]].values
    y = Gapp_info_re.iloc[:, 9:10].values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    y_train = class_le.fit_transform(y_train)
    y_test = class_le.fit_transform(y_test)
    
    rfc_k = RandomForestClassifier()
    rfc_k.fit(X_train, y_train)
    
    y_pred_k = rfc_k.predict(X_test)
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_k, average='micro') 
    
    fscores.append(fscore)
    print("Top", k)
    print(f_importances.index[:k])
    print(precision, recall, fscore, support)


# ## Random Forest for Installation count

# In[36]:


train_score_ins = []
test_score_ins = []
F_score_ins = []

rf_installcount = ['Size_r', 'Category', 'Price_r', 'Content_Rating']

y = Gapp_info_re.Installs_re.values
X = Gapp_info_re.ix[:, rf_installcount].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train_std, y_train)
y_pred = rfc.predict(X_test_std)

train_score_ins.append(rfc.score(X_train_std, y_train))
test_score_ins.append(rfc.score(X_test_std, y_test))
F_score_ins.append(f1_score(y_test, y_pred, average='weighted'))

print('Training accuracy', rfc.score(X_train_std, y_train))
print('Testing accuracy', rfc.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ## KNN for Installation Count

# In[237]:


knn_installcount = ['Size_r', 'Category', 'Content_Rating', 'Price_r']

y = Gapp_info_re.Installs_re.values
X = Gapp_info_re.ix[:, knn_installcount].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 2000, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

n_neighbors = 10

knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

train_score_ins.append(knn.score(X_train_std, y_train))
test_score_ins.append(knn.score(X_test_std, y_test))
F_score_ins.append(f1_score(y_test, y_pred, average='weighted'))

print(knn.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[230]:


from sklearn.model_selection import StratifiedKFold


# In[234]:


cv = StratifiedKFold(n_splits=10, random_state=1000)
n_neighbors = 10

y = Gapp_info_re.Installs_re.values
X = Gapp_info_re.ix[:, knn_installcount].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

score = []
for train_index, test_index in cv.split(X, y):
    X_train_std, X_test_std = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(X_train_std, y_train)
    score.append(clf.score(X_test_std, y_test))
    
score_m = sum(score)/ len(score)
print(score_m)    


# ## Decision Tree

# In[227]:


dt_installcount = ['Size_r', 'Category', 'Content_Rating', 'Price_r']

y = Gapp_info_re.Installs_re.values
X = Gapp_info_re.ix[:, dt_installcount].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 2000, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

dt = tree.DecisionTreeClassifier()
dt.fit(X_train_std,y_train)
y_pred = dt.predict(X_test_std)

train_score_ins.append(dt.score(X_train_std, y_train))
test_score_ins.append(dt.score(X_test_std, y_test))
F_score_ins.append(f1_score(y_test, y_pred, average='weighted'))

print("Decision Tree Testing accuracy: ",dt.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[228]:


Model_ins = ['Random Forest Classifier', 'KNN', 'Decision Tree']
Model_ins_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score', 'F1_score'])

for i in range(3):
    Model_ins_comparison.loc[i, 'Classfier_name'] = Model_ins[i]
    Model_ins_comparison.loc[i, 'train_score'] = train_score_ins[i]
    Model_ins_comparison.loc[i, 'test_score'] = test_score_ins[i]
    Model_ins_comparison.loc[i, 'F1_score'] = F_score_ins[i]
    
Model_ins_comparison


# # Rating Analysis

# In[202]:


plt.figure(figsize=(6,6))
Gapp_info_re.rating_type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Rating >= 4.0 (1) vs <4.0 (0)')
plt.show()


# ## Feature Importance for Rating

# In[114]:


Gapp_info_col = ['Category', 'Price_Type', 'Content_Rating', 'Size_r', 'Price_r']
y = Gapp_info_re.rating_type.values
X_rating = Gapp_info_re.ix[:, Gapp_info_col]
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Random forest classifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train) 

importances = rfc.feature_importances_

# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X_rating.columns[:])

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

f_importances


# In[115]:


fscores = []
for k in range(1, 6):
    X = X_df[f_importances.index[:k]].values
    y = Gapp_info_re.iloc[:, 13:14].values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    y_train = class_le.fit_transform(y_train)
    y_test = class_le.fit_transform(y_test)
    
    rfc_k = RandomForestClassifier(random_state=0)
    rfc_k.fit(X_train, y_train)
    
    y_pred_k = rfc_k.predict(X_test)
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_k, average='micro') 
    
    fscores.append(fscore)
    print("Top", k)
    print(f_importances.index[:k])
    print(precision, recall, fscore, support)


# ## Random Forest for Rating

# In[211]:


train_score = []
test_score = []
F_score = []

Gapp_info_col = ['Size_r', 'Category', 'Price_r']

y = Gapp_info_re.rating_type.values
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 1000, stratify=y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train_std, y_train)
y_pred = rfc.predict(X_test_std)

train_score.append(rfc.score(X_train_std, y_train))
test_score.append(rfc.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average='weighted'))

print('Training accuracy', rfc.score(X_train_std, y_train))
print('Testing accuracy', rfc.score(X_test_std, y_test))
print(classification_report(y_test, y_pred))


# In[181]:


Model = []
Model.append('Random Forest Classification')
AUC_all = []
Fpr = []
Tpr = []

rf_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(rf_roc_auc)
fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)
plt.figure()
plt.plot(fpr_f, tpr_f, label='Random Forest Classification (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating')
plt.legend(loc="lower right")

plt.show()


# ## KNN for Rating

# In[212]:


knn_rating = ['Size_r', 'Category', 'Price_r']

y = Gapp_info_re.rating_type.values
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1000, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

n_neighbors = 5

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

train_score.append(knn.score(X_train_std, y_train))
test_score.append(knn.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average='weighted'))

print('Testing accuracy',knn.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[183]:


Model.append('KNN')
knn_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(knn_roc_auc)
fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)
plt.figure()
plt.plot(fpr_f, tpr_f, label='KNN Classification (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating')
plt.legend(loc="lower right")

plt.show()


# ## Logistic Regression for Rating

# In[213]:


lr_rating = ['Size_r', 'Category', 'Price_r']

y = Gapp_info_re.rating_type.values
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_std,y_train)
y_pred = lr.predict(X_test_std)

train_score.append(lr.score(X_train_std, y_train))
test_score.append(lr.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average='weighted'))

print("Logistic Regression Testing accuracy: ",lr.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[205]:


lr_rating = ['Size_r', 'Category', 'Price_r']

y = Gapp_info_re.rating_type.values
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_std,y_train)
y_pred = lr.predict(X_test_std)

print(f1_score(y_test, y_pred, average='weighted'))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred))


# In[185]:


Model.append('Logistic Regression')
lr_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(lr_roc_auc)
fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)
plt.figure()
plt.plot(fpr_f, tpr_f, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating')
plt.legend(loc="lower right")

plt.show()


# ## Decision Tree

# In[214]:


dt_rating = ['Size_r', 'Category', 'Price_r']

y = Gapp_info_re.rating_type.values
X = Gapp_info_re.ix[:, Gapp_info_col].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

dt = tree.DecisionTreeClassifier()
dt.fit(X_train_std,y_train)
y_pred = dt.predict(X_test_std)

train_score.append(dt.score(X_train_std, y_train))
test_score.append(dt.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average='weighted'))

print("Decision Tree Testing accuracy: ",dt.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[204]:


print(f1_score(y_test, y_pred, average='weighted'))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred))


# In[187]:


Model.append('Decision Tree')
dt_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(dt_roc_auc)
fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)
plt.figure()
plt.plot(fpr_f, tpr_f, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating')
plt.legend(loc="lower right")

plt.show()


# ### Model Comparison

# In[215]:


Model_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score', 'F1_score'])

for i in range(4):
    Model_comparison.loc[i, 'Classfier_name'] = Model[i]
    Model_comparison.loc[i, 'train_score'] = train_score[i]
    Model_comparison.loc[i, 'test_score'] = test_score[i]
    Model_comparison.loc[i, 'F1_score'] = F_score[i]
    
Model_comparison


# ### AUC ROC Comparison

# In[189]:


plt.figure(figsize=(12,8))

for i in range(4):
    plt.plot(Fpr[i], Tpr[i], label = Model[i] + '(area = %0.2f)' % AUC_all[i])
    
plt.plot([0, 1], [0, 1],'p--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating')
plt.legend(loc="lower right")

plt.show() 

