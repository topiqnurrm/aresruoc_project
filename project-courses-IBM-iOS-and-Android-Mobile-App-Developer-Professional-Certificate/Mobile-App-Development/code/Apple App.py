
# coding: utf-8

# In[312]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


app_info = pd.read_csv('AppleStore.csv')


# In[287]:


app_info.head()


# In[4]:


app_info.columns


# In[4]:


# remove and rename columns 
app_info = app_info.drop(columns = ['Unnamed: 0', 'id'], axis = 1)
app_info.rename(columns = {'rating_count_ver':'rating_count_cur','user_rating_ver':'user_rating_cur','ver':'version' , 'sup_devices.num':'sup_devices_num','ipadSc_urls.num':'screenshot_num', 'lang.num':'lang_num'}, inplace = True)
app_info.head()


# In[6]:


# check null value
print('App_info null value:')
print(app_info.isnull().sum())


# In[7]:


# check duplicated data

print('Duplicate data in app info:')
print(app_info.duplicated().sum())


# In[8]:


plt.figure(figsize=(14,8))
app_info.price.value_counts().plot(kind = 'bar')
plt.title('App Price')
plt.show()


# In[376]:


plt.figure(figsize=(10,6))
app_info.user_rating_cur.value_counts().sort_index().plot(kind = 'bar')
plt.title('App Rating(Current version)')
plt.show()


# In[5]:


# drop rating = 0.0
app_info = app_info[app_info.user_rating != 0.0]


# In[375]:


plt.figure(figsize=(10,6))
app_info.user_rating.value_counts().sort_index().plot(kind = 'bar')
plt.title('App Rating')
plt.show()


# In[329]:


plt.figure(figsize=(16,8))
plt.plot(app_info.rating_count_tot)
plt.plot(app_info.rating_count_cur)
plt.title('Rating Count (Total vs Current Version)')
plt.legend()
plt.show()


# In[40]:


plt.figure(figsize=(6,6))
app_info.cont_rating.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('App Content Rating')
plt.show()


# In[6]:


# replace string (1: all age, 2: age over9, 3: age over 12, 4: age over 17)
app_info_re = app_info
app_info_re = app_info_re.replace('4+',1)
app_info_re = app_info_re.replace('9+',2)
app_info_re = app_info_re.replace('12+',3)
app_info_re = app_info_re.replace('17+',4)


# In[136]:


plt.figure(figsize=(12,8))
app_info.prime_genre.value_counts().plot(kind = 'bar')
plt.title('App Genre')
plt.show()


# In[7]:


# replace string
app_info_re = app_info_re.replace('Games',1)
app_info_re = app_info_re.replace('Entertainment',2)
app_info_re = app_info_re.replace('Education',3)
app_info_re = app_info_re.replace('Photo & Video',4)
app_info_re = app_info_re.replace('Utilities',5)
app_info_re = app_info_re.replace('Health & Fitness',6)
app_info_re = app_info_re.replace('Productivity',7)
app_info_re = app_info_re.replace('Social Networking',8)
app_info_re = app_info_re.replace('Lifestyle',9)
app_info_re = app_info_re.replace('Music',10)
app_info_re = app_info_re.replace('Shopping',11)
app_info_re = app_info_re.replace('Sports',12)
app_info_re = app_info_re.replace('Book',13)
app_info_re = app_info_re.replace('Finance',14)
app_info_re = app_info_re.replace('Travel',15)
app_info_re = app_info_re.replace('News',16)
app_info_re = app_info_re.replace('Weather',17)
app_info_re = app_info_re.replace('Reference',18)
app_info_re = app_info_re.replace('Food & Drink',19)
app_info_re = app_info_re.replace('Business',20)
app_info_re = app_info_re.replace('Navigation',21)
app_info_re = app_info_re.replace('Medical',22)
app_info_re = app_info_re.replace('Catalogs',23)


# In[16]:


app_info.head()


# In[8]:


app_info_re = app_info_re.drop(columns = ['currency', 'version', 'vpp_lic'], axis = 1)


# In[18]:


plt.figure(figsize=(10,8))
app_info.sup_devices_num.value_counts().plot(kind = 'bar')
plt.title('Number of Supporting Device')
plt.show()


# In[19]:


plt.figure(figsize=(6,6))
app_info.screenshot_num.value_counts().plot(kind = 'bar')
plt.title('Number of Screenshot')
plt.show()


# In[20]:


plt.figure(figsize=(16,8))
app_info.lang_num.value_counts().plot(kind = 'bar')
plt.title('Number of Language')
plt.show()


# In[9]:


app_info_re['size_bytes'] = app_info_re['size_bytes'].apply(float)
app_info_re['price'] = app_info_re['price'].apply(float)
app_info_re['rating_count_tot'] = app_info_re['rating_count_tot'].apply(int)
app_info_re['rating_count_cur'] = app_info_re['rating_count_cur'].apply(int)
app_info_re['user_rating'] = app_info_re['user_rating'].apply(float)
app_info_re['user_rating_cur'] = app_info_re['user_rating_cur'].apply(float)
app_info_re['prime_genre'] = app_info_re['prime_genre'].apply(int)
app_info_re['sup_devices_num'] = app_info_re['sup_devices_num'].apply(int)
app_info_re['screenshot_num'] = app_info_re['screenshot_num'].apply(int)
app_info_re['lang_num'] = app_info_re['lang_num'].apply(int)
app_info_re['cont_rating'] = app_info_re['cont_rating'].apply(int)


# In[10]:


app_info_free = app_info_re[app_info_re.price == 0.0]
app_info_free.head()


# In[289]:


price_type = []
for i in app_info_re.price:
    if i == 0.00:
        price_type.append(0)
    else:
        price_type.append(1)


# In[288]:


Price_type = pd.DataFrame(price_type, columns = ['price_type'], index = app_info_re.index)
Price_type.head()


# In[13]:


app_info_re['price_type'] = Price_type


# In[27]:


plt.figure(figsize=(6,6))
app_info_re.price_type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Pay or Not')
plt.show()


# ### Add A Column for rating type useing current rating

# In[14]:


rating = []
for i in app_info_re.user_rating_cur:
    if i < 4.0:
        rating.append(0)
    else:
        rating.append(1)
        
app_info_re['rating_type'] = rating


# In[29]:


plt.figure(figsize=(6,6))
app_info_re.rating_type.value_counts().plot(kind = 'bar')
plt.title('Rating Type')
plt.show()


# In[15]:


app_info_re['rating_count_be'] = app_info_re['rating_count_tot'] - app_info_re['rating_count_cur']


# In[17]:


app_info_re.head()


# In[19]:


#correlation map
cols = ['size_bytes', 'price', 'rating_count_tot', 'rating_count_cur', 'user_rating','user_rating_cur','cont_rating', 'prime_genre','sup_devices_num', 'screenshot_num', 'lang_num']

cm = np.corrcoef(app_info_re[cols].values.T)

# Get heat map
plt.figure(figsize=(12,12))
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

# Show heat map
plt.tight_layout()
plt.show()


# In[122]:


# Pair Grid - showing the relationships in the dataset
df_plot = app_info_re.ix[:, cols]

pairplot = sns.PairGrid(df_plot, hue = 'user_rating', palette = "Blues")
pairplot = pairplot.map(plt.scatter)
pairplot = pairplot.add_legend()


# In[34]:


# Rating Count for Total
col_cur = ['size_bytes', 'price', 'price_type','rating_count_tot', 'user_rating','cont_rating','sup_devices_num', 'screenshot_num', 'lang_num']
target_idx = len(col_cur) - 6
for exp_var_idx in range(len(col_cur)):
    
    sns.lmplot(x = app_info_re[col_cur].columns[exp_var_idx], y = app_info_re[col_cur].columns[target_idx], data = app_info_re[col_cur], line_kws={"color":"red"})  
    
plt.tight_layout()
plt.show()


# In[35]:


# Price vs. Geners
pd.crosstab(app_info.prime_genre,app_info_re.price_type).plot(kind='bar', figsize = (14,8))
plt.title('Price vs. Geners')
plt.xlabel('Geners')
plt.ylabel('Count')
plt.show()


# In[36]:


# Price vs. Rating
pd.crosstab(app_info.user_rating,app_info_re.price_type).plot(kind='bar')
plt.title('Price vs. Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[37]:


# Price vs. Current Rating
pd.crosstab(app_info.user_rating_cur,app_info_re.price_type).plot(kind='bar')
plt.title('Price vs. Current Rating')
plt.xlabel('Current Rating')
plt.ylabel('Count')
plt.show()


# In[38]:


# Price vs. Content Rating
pd.crosstab(app_info.cont_rating,app_info_re.price_type).plot(kind='bar')
plt.title('Price vs. Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Count')
plt.show()


# In[39]:


app_info_re.head()


# ## Feature Importance

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve


# In[82]:


plt.figure(figsize=(6,6))
app_info_re.rating_type.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Rating >= 4.0 (1) vs <4.0 (0)')
plt.show()


# ### Current Rating's feature importance

# In[204]:


fm_rating = ['size_bytes', 'price', 'price_type','rating_count_be','cont_rating','sup_devices_num', 'screenshot_num', 'lang_num', 'prime_genre']

y = app_info_re.iloc[:, 13:14].values.reshape(-1, 1)
X_df = app_info_re.ix[:, fm_rating]
X = app_info_re.ix[:, fm_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[205]:


# Random forest classifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train) 

importances = rfc.feature_importances_
# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X_df.columns[:])

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

f_importances


# In[206]:


fscores = []
for k in range(1, 10):
    X = X_df[f_importances.index[:k]].values
    y = app_info_re.iloc[:, 13:14].values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)
    
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


# ## Random Forest for Current Rating

# In[248]:


train_score = []
test_score = []
F_score = []

rf_rating = ['rating_count_be', 'size_bytes', 'prime_genre', 'lang_num', 'sup_devices_num']

y = app_info_re.rating_type.values
X = app_info_re.ix[:, rf_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

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


# In[249]:


print('Random Forest Training accuracy', rfc.score(X_train_std, y_train))
print('Random Forest Tseting accuracy', rfc.score(X_test_std, y_test))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[250]:


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


# ## KNN for User Current Rating

# In[251]:


knn_rating = ['rating_count_be', 'lang_num', 'prime_genre','sup_devices_num', 'size_bytes']

y = app_info_re.rating_type.values
X = app_info_re.ix[:, knn_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

n_neighbors = 10

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

train_score.append(knn.score(X_train_std, y_train))
test_score.append(knn.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average='weighted'))

print('KNN Training Accuracy',knn.score(X_train_std, y_train))
print('KNN Testing Accuracy',knn.score(X_test_std, y_test))
print('F1 score', f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))


# In[252]:


Model.append('KNN')
knn_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(knn_roc_auc)

fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)


# ### Try PCA Feature Selection for Rating
# #### To see if get the better accuracy

# In[165]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[224]:


app_info_co = ['size_bytes', 'cont_rating', 'prime_genre', 'screenshot_num','lang_num', 'rating_count_be', 'sup_devices_num']

y = app_info_re.rating_type.values
X = app_info_re.ix[:, app_info_co].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(X)

pca = PCA().fit(data_rescaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.show()


# In[225]:


pca = PCA(n_components=5)
X = pca.fit_transform(data_rescaled)

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

print('KNN Training Accuracy(after PCA)', knn.score(X_train_std, y_train))
print('KNN Testing Accuracy(after PCA)', knn.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### Cross Validation 

# In[168]:


from sklearn.model_selection import StratifiedKFold


# In[169]:


#from sklearn.model_selection import cross_val_score
cv = StratifiedKFold(n_splits=10, random_state=2000)
n_neighbors = 10

y = app_info_re.rating_type.values
X = app_info_re.ix[:, knn_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

score = []
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(X_train, y_train)
    score.append(clf.score(X_test, y_test))
    
score_m = sum(score)/ len(score)
print(score_m)    


# ## Logistic Regression for User Current Rating

# In[253]:


lr_rating = ['rating_count_be', 'lang_num', 'prime_genre', 'size_bytes', 'sup_devices_num']

y = app_info_re.rating_type.values
X = app_info_re.ix[:, lr_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

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

print("Logistic Regression Training accuracy: ",lr.score(X_train_std,y_train))
print("Logistic Regression Testing accuracy: ",lr.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[254]:


Model.append('Logistic Regression')
lr_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(lr_roc_auc)

fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)


# ## Decision Tree For User Current Rating

# In[258]:


dt_rating = ['rating_count_be', 'lang_num', 'prime_genre', 'size_bytes', 'sup_devices_num']

y = app_info_re.rating_type.values
X = app_info_re.ix[:, dt_rating].values

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

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

print("Decision Tree Training accuracy: ",dt.score(X_train_std,y_train))
print("Decision Tree Testing accuracy: ",dt.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[256]:


Model.append('Decision Tree')
dt_roc_auc = roc_auc_score(y_test, y_pred)
AUC_all.append(dt_roc_auc)

fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred)
Fpr.append(fpr_f)
Tpr.append(tpr_f)


# ### Model Comparison

# In[257]:


Model_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score', 'F1_score'])

for i in range(4):
    Model_comparison.loc[i, 'Classfier_name'] = Model[i]
    Model_comparison.loc[i, 'train_score'] = train_score[i]
    Model_comparison.loc[i, 'test_score'] = test_score[i]
    Model_comparison.loc[i, 'F1_score'] = F_score[i]
    
Model_comparison


# ### Comparison of AUC & ROC Curve 

# In[45]:


plt.figure(figsize=(12,8))
for i in range(4):
    
    plt.plot(Fpr[i], Tpr[i], label = Model[i] + '(area = %0.2f)' % AUC_all[i])
        
plt.plot([0, 1], [0, 1],'p--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('App User rating ROC')
plt.legend(loc="lower right")

plt.show() 


# # Rating Count's Regression

# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# ## Linear Regression 

# In[309]:


'''model_count = []
train_acc = []
test_acc = []'''

lr_ratingcount = ['size_bytes','rating_count_be','cont_rating','lang_num', 'sup_devices_num']

y = app_info_re.iloc[:, 3:4].values.reshape(-1, 1)
X = app_info_re.ix[:, lr_ratingcount].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

'''sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)'''

regression = LinearRegression()
regression.fit(X_train, y_train)

model_count.append('Linear Regression')
train_acc.append(regression.score(X_train, y_train))
test_acc.append(regression.score(X_test, y_test))


# In[310]:


y_pred = regression.predict(X_test)
print('Linear Regression R Square', regression.score(X_test, y_test))

mse = mean_squared_error(y_pred, y_test)
print('Linear Regression MSE', mse)


# ## Random Forest Regression

# In[285]:


rfr_ratingcount = ['size_bytes', 'price', 'price_type','rating_count_be','cont_rating','sup_devices_num', 'screenshot_num', 'lang_num', 'prime_genre']

y = app_info_re.rating_count_tot.values
X = app_info_re.ix[:, rfr_ratingcount].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rf_reg = RandomForestRegressor(random_state = 0)
rf_reg.fit(X_train_std, y_train)
y_pred = rf_reg.predict(X_test_std)

X_df = app_info_re.ix[:, rfr_ratingcount]
importances = rf_reg.feature_importances_
# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X_df.columns[:])

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

f_importances


# In[286]:


rfr_ratingcount = ['size_bytes','rating_count_be','cont_rating','lang_num', 'sup_devices_num']

y = app_info_re.rating_count_tot.values
X = app_info_re.ix[:, rfr_ratingcount].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_std, y_train)
y_pred = rf_reg.predict(X_test_std)

model_count.append('Random Forest Regressor')
train_acc.append(rf_reg.score(X_train_std, y_train))
test_acc.append(rf_reg.score(X_test_std, y_test))

print('Ramdon Forest R Square', rf_reg.score(X_test_std, y_test))


# In[273]:


Model_comparison_count = pd.DataFrame(columns=['Regression_name', 'train_score', 'R square'])

for i in range(2):
    Model_comparison_count.loc[i, 'Regression_name'] = model_count[i]
    Model_comparison_count.loc[i, 'train_score'] = train_acc[i]
    Model_comparison_count.loc[i, 'R square'] = test_acc[i]
    
Model_comparison_count


# ### Rating more than 4.0

# In[63]:


top_rating = app_info[app_info['user_rating'] >= 4.0]
top_rating['price_type'] = app_info_re['price_type']
top_rating


# In[64]:


plt.figure(figsize=(14,8))
top_rating.prime_genre.value_counts().plot(kind = 'bar')
plt.title('Top Rating Apps Genre')
plt.show()


# In[65]:


pd.crosstab(top_rating.prime_genre,top_rating.price_type).plot(kind='bar', figsize = (14,8))
plt.title('Price Type vs. Prime Genre')
plt.xlabel('Prime Genre')
plt.ylabel('Count')
plt.show()


# In[66]:


pd.crosstab(top_rating.cont_rating,top_rating.price_type).plot(kind='bar', figsize = (10,8))
plt.title('Price Type vs. Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Count')
plt.show()


# ### Sort By Rating Count
# 
# 

# In[68]:


# Sort by rating count

Rating_count = app_info.sort_values('rating_count_tot',ascending = False).head(100)
Rating_count['price_type'] = app_info_re['price_type']
Rating_count


# In[69]:


plt.figure(figsize=(8,8))
Rating_count.user_rating.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Top 100 Installation App Rating(All Version)')
plt.show()


# In[70]:


plt.figure(figsize=(8,8))
Rating_count.price.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Top 100 Installation App Price')
plt.show()


# In[71]:


plt.figure(figsize=(14,8))
Rating_count.prime_genre.value_counts().plot(kind = 'bar')
plt.title('Top 100 Installation Apps Genre')
plt.show()


# In[72]:


pd.crosstab(Rating_count.prime_genre,Rating_count.price_type).plot(kind='bar', figsize = (14,8))
plt.title('Price Type vs. Prime Genre')
plt.xlabel('Prime Genre')
plt.ylabel('Count')
plt.show()


# In[73]:


pd.crosstab(Rating_count.cont_rating,Rating_count.price_type).plot(kind='bar', figsize = (10,8))
plt.title('Price Type vs. Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Count')
plt.show()

