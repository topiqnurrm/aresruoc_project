
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


Gapp_review = pd.read_csv('googleplaystore_user_reviews.csv')
Gapp_info = pd.read_csv('googleplaystore.csv')


# ## App Reviews

# In[136]:


# check null value
Gapp_review.isnull().sum()


# In[23]:


# remove Translated_Review = null
Gapp_review.dropna(subset=['Translated_Review'], inplace = True)
Gapp_review.head()


# In[24]:


# Check duplicate
print(Gapp_review.duplicated().sum())
Gapp_review.drop_duplicates(keep = 'first', inplace = True)
Gapp_review.reset_index(inplace = True, drop = True)
Gapp_review.head()


# In[6]:


plt.figure(figsize=(6,6))
Gapp_review.Sentiment.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('App Reviews Sentiment')
plt.show()


# In[61]:


# plot the sentiment
plt.figure(figsize=(18,16))
plt.scatter(Gapp_review['Sentiment_Polarity'],Gapp_review['Sentiment_Subjectivity'])
plt.xlabel('Sentiment Polarity', fontsize=20)
plt.ylabel('Sentiment Subjectivity', fontsize=20)
plt.title('Sentiment Plot', fontsize=20)
plt.show()


# ## App Information

# In[25]:


# rename 
Gapp_info.rename(columns = {'Type': 'Price_Type','Content Rating': 'Content_Rating', 'Last Updated':'Last_Updated', 'Current Ver': 'Current_Ver', 'Android Ver':'Android_Ver'}, inplace= True)
Gapp_info.head()


# In[27]:


# remove rating = null
Gapp_info.dropna(subset=['Rating'], inplace = True)
Gapp_info.drop_duplicates(keep = 'first', inplace = True)

Gapp_info.head()


# In[28]:


Gapp_info_re = Gapp_info.drop(columns = ['Genres', 'Last_Updated', 'Current_Ver','Android_Ver'])
Gapp_info_re.drop_duplicates(keep = 'first', inplace = True)
Gapp_info_re.head()


# In[48]:


Gapp_info_re = Gapp_info_re[Gapp_info_re.Rating != 19.0]
Gapp_info_re = Gapp_info_re[Gapp_info_re.Price != 'Everyone']
Gapp_info_re = Gapp_info_re[Gapp_info_re.Category != '1.9']
Gapp_info_re = Gapp_info_re[Gapp_info_re.Installs != 'Free']
Gapp_info_re.drop_duplicates(subset = 'App', keep = 'first', inplace = True)
Gapp_info_re.reset_index(inplace = True, drop = True)


# In[49]:


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


# In[50]:


Gapp_info_re.head()


# # The Most Frequent Words for the App Reviews

# In[31]:


Gapp_review_only = Gapp_review.drop(columns = ['Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity'], axis = 1)
Gapp_review_only.head()


# In[32]:


# combine the reviews for the same app
App = Gapp_review_only.groupby('App')
Gapp_review_only = App.sum()
Gapp_review_only.reset_index(inplace = True, drop = False)
Gapp_review_only.head()


# In[14]:


# Find the top three most frequent words
'''
for i in range(Gapp_review_only.shape[0]):
    words = Gapp_review_only['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Gapp_review_only.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Gapp_review_only.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Gapp_review_only.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        Gapp_review_only.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Gapp_review_only.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    else:
        Gapp_review_only.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
        
Gapp_review_only.to_csv('Gapp_review_only.csv')        
'''        


# In[33]:


Gapp_review_only = pd.read_csv('Gapp_review_only.csv', index_col = 0)
Gapp_review_only.head()


# In[17]:


Gapp_review_most_freq1 = Gapp_review_only.most_freq_word_1st.value_counts()
Gapp_review_most_freq1 = Gapp_review_most_freq1.drop('i')
Gapp_review_most_freq1 = Gapp_review_most_freq1.drop('it')
Gapp_review_most_freq1 = Gapp_review_most_freq1.drop('the')


# In[18]:


plt.figure(figsize=(14,8))
plt.title('The Most Frequent Words')
Gapp_review_most_freq1.head(20).plot.bar()


# In[117]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Gapp_review_only.most_freq_word_2nd.value_counts().head(20).plot.bar()


# In[118]:


plt.figure(figsize=(16,8))
plt.title('The Third Most Frequent Words')
Gapp_review_only.most_freq_word_3rd.value_counts().head(20).plot.bar()


# #### Word Cloud for all the reviews

# In[18]:


All_re = []
for i in range(Gapp_review_only.shape[0]):
    reviews = Gapp_review_only['Translated_Review'][i]
    All_re.append(reviews)

#Search for all non-letters, Replace all non-letters with spaces
All_re = re.sub("[^a-zA-Z]",  " ", str(All_re))


# In[19]:


# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(All_re)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Separate the Positive, Negative and Neutral reviews

# ### Positive

# In[35]:


# Positive Reviews

Positive = Gapp_review[Gapp_review['Sentiment'] == 'Positive'] 
Positive.reset_index(inplace = True, drop = True)


# In[36]:


Positive.head()


# In[22]:


# Find the top three most frequent words
'''
for i in range(Positive.shape[0]):
    words = Positive['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Positive.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Positive.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Positive.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        Positive.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Positive.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] == 1:
        Positive.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else:
        Positive.loc[i, 'most_freq_word_1st'] = 'NAN'
'''        


# In[37]:


#Positive.to_csv('Positive.csv')
Positive = pd.read_csv('Positive.csv', index_col=0)
Positive.head()


# In[22]:


Positive_most_freq1 = Positive.most_freq_word_1st.value_counts()
Positive_most_freq1 = Positive_most_freq1.drop('i')
Positive_most_freq1 = Positive_most_freq1.drop('it')
Positive_most_freq1 = Positive_most_freq1.drop('the')


# In[89]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
Positive_most_freq1.head(20).plot.bar()


# In[120]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Positive.most_freq_word_2nd.value_counts().head(20).plot.bar()


# #### Word Cloud for positive reviews

# In[38]:


Positive_re = []
for i in range(Positive.shape[0]):
    reviews = Positive['Translated_Review'][i]
    Positive_re.append(reviews)

#Search for all non-letters, Replace all non-letters with spaces
Positive_re = re.sub("[^a-zA-Z]",  " ", str(Positive_re))

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Positive_re)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Neutral

# In[38]:


# Neutral Reviews 

Neutral = Gapp_review[Gapp_review['Sentiment'] == 'Neutral'] 
Neutral.reset_index(inplace = True, drop = True)
Neutral.head()


# In[26]:


# Find the top three most frequent words
'''
for i in range(Neutral.shape[0]):
    words = Neutral['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Neutral.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Neutral.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Neutral.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        Neutral.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Neutral.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] == 1:
        Neutral.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else:
        Neutral.loc[i, 'most_freq_word_1st'] = 'NAN'
'''      


# In[39]:


#Neutral.to_csv('Neutral.csv')
Neutral = pd.read_csv('Neutral.csv', index_col=0)
Neutral.head()


# In[25]:


Neutral_most_freq1 = Neutral.most_freq_word_1st.value_counts()
Neutral_most_freq1 = Neutral_most_freq1.drop('i')
Neutral_most_freq1 = Neutral_most_freq1.drop('it')
Neutral_most_freq1 = Neutral_most_freq1.drop('yet')


# In[110]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
Neutral_most_freq1.head(30).plot.bar()


# In[121]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Neutral.most_freq_word_2nd.value_counts().head(20).plot.bar()


# #### Word Cloud for Neutral reviews

# In[39]:


Neutral_re = []
for i in range(Neutral.shape[0]):
    reviews = Neutral['Translated_Review'][i]
    Neutral_re.append(reviews)

#Search for all non-letters, Replace all non-letters with spaces
Neutral_re = re.sub("[^a-zA-Z]",  " ", str(Neutral_re))

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Neutral_re)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Negative

# In[40]:


# Negative Reviews

Negative = Gapp_review[Gapp_review['Sentiment'] == 'Negative'] 
Negative.reset_index(inplace = True, drop = True)
Negative.head()


# In[32]:


# Find the top three most frequent words
'''
for i in range(Negative.shape[0]):
    words = Negative['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Negative.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Negative.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Negative.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        Negative.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Negative.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] == 1:
        Negative.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else:
        Negative.loc[i, 'most_freq_word_1st'] = 'NAN'
'''       


# In[41]:


#Negative.to_csv('Negative.csv')
Negative = pd.read_csv('Negative.csv', index_col=0)
Negative.head()


# In[20]:


Negative_most_freq1 = Negative.most_freq_word_1st.value_counts()
Negative_most_freq1 = Negative_most_freq1.drop('i')
Negative_most_freq1 = Negative_most_freq1.drop('it')
Negative_most_freq1 = Negative_most_freq1.drop('the')
Negative_most_freq1 = Negative_most_freq1.drop('u')


# In[21]:


plt.figure(figsize=(14,8))
plt.title('The Most Frequent Words')
Negative_most_freq1.head(30).plot.bar()


# In[122]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Negative.most_freq_word_2nd.value_counts().head(20).plot.bar()


# #### Word Cloud for Negative Reviews

# In[41]:


Negative_re = []
for i in range(Negative.shape[0]):
    reviews = Negative['Translated_Review'][i]
    Negative_re.append(reviews)

#Search for all non-letters, Replace all non-letters with spaces
Negative_re = re.sub("[^a-zA-Z]",  " ", str(Negative_re))

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Negative_re)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## The most frequent words in the Facebook reviews

# In[42]:


Gapp_review.head()


# In[43]:


Facebook = Gapp_review[Gapp_review['App'] == 'Facebook'] 
Facebook.reset_index(inplace = True, drop = True)
Facebook.head()


# In[125]:


for i in range(Facebook.shape[0]):
    words = Facebook['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Facebook.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Facebook.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Facebook.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        Facebook.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Facebook.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    else:
        Facebook.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]


# In[127]:


Facebook_most_freq1 = Facebook.most_freq_word_1st.value_counts()
Facebook_most_freq1 = Facebook_most_freq1.drop('i')


# In[128]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
Facebook_most_freq1.head(20).plot.bar()


# In[129]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Facebook.most_freq_word_2nd.value_counts().head(20).plot.bar()


# In[43]:


plt.figure(figsize=(6,6))
Facebook.Sentiment.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Facebook Reviews Sentiment')
plt.show()


# In[79]:


Facebook_re = []
for i in range(Facebook.shape[0]):
    review = Facebook['Translated_Review'][i]
    Facebook_re.append(review)

#Search for all non-letters, Replace all non-letters with spaces
Facebook_re = re.sub("[^a-zA-Z]",  " ", str(Facebook_re))


# #### Word Cloud for all the reviews

# In[87]:


# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Facebook_re)
# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[44]:


Episode = Gapp_review[Gapp_review['App'] == 'Episode - Choose Your Story'] 
Episode.reset_index(inplace = True, drop = True)
Episode.head()


# In[93]:


for i in range(Episode.shape[0]):
    words = Episode['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    Episode.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]


# In[94]:


Episode


# In[56]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
Episode.most_freq_word_1st.value_counts().head(20).plot.bar()


# In[57]:


plt.figure(figsize=(6,6))
Episode.Sentiment.value_counts().plot(kind = 'pie', autopct='%.2f', fontsize=12)
plt.title('Episode Reviews Sentiment')
plt.show()


# In[97]:


Episode_re = []
for i in range(Episode.shape[0]):
    review = Episode['Translated_Review'][i]
    Episode_re.append(review)

#Search for all non-letters, Replace all non-letters with spaces
Episode_re = re.sub("[^a-zA-Z]",  " ", str(Episode_re))


# #### Word Cloud for all the reviews

# In[98]:


# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Episode_re)
# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Top 500 installation Count

# In[51]:


Install_count = Gapp_info_re.sort_values('Installs',ascending = False).head(500)
Install_count.reset_index(inplace = True, drop = True)
Install_count.head(10)


# In[52]:


Install_count_top500 = Gapp_review.loc[Gapp_review['App'].isin(Install_count['App'])]
Install_count_top500.reset_index(inplace = True, drop = True)
Install_count_top500.head()


# In[135]:


# Find the top three most frequent words
'''
for i in range(Install_count_top500.shape[0]):
    words = Install_count_top500['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # convert words to stem
    lemma = nltk.WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        Install_count_top500.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Install_count_top500.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        Install_count_top500.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        A.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        Install_count_top500.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] == 1:
        Install_count_top500.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else:
        Install_count_top500.loc[i, 'most_freq_word_1st'] = 'NAN'
        
Install_count_top500.to_csv('Install_count_top500.csv')        
'''


# In[53]:


Install_count_top500 = pd.read_csv('Install_count_top500.csv', index_col=0)
Install_count_top500.head()


# In[138]:


instalcount_most_freq1 = Install_count_top500.most_freq_word_1st.value_counts()
instalcount_most_freq1 = instalcount_most_freq1.drop('i')
instalcount_most_freq1 = instalcount_most_freq1.drop('it')
instalcount_most_freq1 = instalcount_most_freq1.drop('the')
instalcount_most_freq1 = instalcount_most_freq1.drop('this')


# In[140]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
instalcount_most_freq1.head(20).plot.bar()


# In[101]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
Install_count_top500.most_freq_word_2nd.value_counts().head(20).plot.bar()


# #### Word Cloud for all the reviews

# In[98]:


Install_count_m_re = []
for i in range(Install_count_top500.shape[0]):
    review = Install_count_top500['Translated_Review'][i]
    Install_count_m_re.append(review)

#Search for all non-letters, Replace all non-letters with spaces
Install_count_m_re = re.sub("[^a-zA-Z]",  " ", str(Install_count_m_re))


# In[99]:


# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(Install_count_m_re)
# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Top Rating App reviews

# In[54]:


toprating = Gapp_info_re[Gapp_info_re['Rating'] >= 4.0]
toprating.reset_index(inplace = True, drop = True)
toprating.head()


# In[55]:


R = Gapp_review.loc[Gapp_review['App'].isin(Install_count['App'])]
R.reset_index(inplace = True, drop = True)
R.head()


# In[80]:


# Find the top three most frequent words
'''
for i in range(R.shape[0]):
    words = R['Translated_Review'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        R.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        R.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        R.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        R.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]    
        R.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] == 1:
        R.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else:
        R.loc[i, 'most_freq_word_1st'] = 'NAN'
    '''


# In[56]:


#R.to_csv('toprating_reviews.csv')
toprating_reviews = pd.read_csv('toprating_reviews.csv', index_col = 0)
toprating_reviews.head()


# In[88]:


toprating_most_freq1 = toprating_reviews.most_freq_word_1st.value_counts()
toprating_most_freq1 = toprating_most_freq1.drop('i')
toprating_most_freq1 = toprating_most_freq1.drop('it')
toprating_most_freq1 = toprating_most_freq1.drop('the')
toprating_most_freq1 = toprating_most_freq1.drop('this')


# In[89]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
toprating_most_freq1.head(20).plot.bar()


# In[100]:


toprating_re = []
for i in range(toprating_reviews.shape[0]):
    reviews = toprating_reviews['Translated_Review'][i]
    toprating_re.append(reviews)

#Search for all non-letters, Replace all non-letters with spaces
toprating_re = re.sub("[^a-zA-Z]",  " ", str(toprating_re))

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(toprating_re)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Classification

# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from scipy import interp
from itertools import cycle


# In[12]:


all_reviews = []
for i in Gapp_review.Translated_Review:
    words = re.sub("[^a-zA-Z]"," ",i)
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # convert words to stem
    lemma = nltk.WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
    text = " ".join(word_list)
    all_reviews.append(text)


# In[13]:


max_features = 1000
cou_vec = CountVectorizer(max_features=max_features)
text_matrix = cou_vec.fit_transform(all_reviews).toarray()
all_words = cou_vec.get_feature_names()
print("Most used 50 words: ",all_words[0:30])


# In[14]:


apps = []
for i in Gapp_review.App:
    words = re.sub("[^a-zA-Z0-9]"," ",i)
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    app_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    app_list = [word.lower() for word in app_list if word not in stopwords.words('english')] 
    # convert words to stem
    lemma = nltk.WordNetLemmatizer()
    app_list = [lemma.lemmatize(word) for word in app_list]
    text = " ".join(app_list)
    apps.append(text)


# In[15]:


max_features = 1000
cou_vec = CountVectorizer(max_features=max_features)
app_matrix = cou_vec.fit_transform(apps).toarray()


# ### Convert Positive, Neutral, Negative into  1, 0, -1

# In[33]:


Sentiment_re = []

for i in Gapp_review.Sentiment:
    if i == 'Positive':
        Sentiment_re.append(1)
    elif i == 'Neutral':
        Sentiment_re.append(0)
    else:
        Sentiment_re.append(-1)
        
Gapp_review['Sentiment_class'] = Sentiment_re
Gapp_review.head()        


# ## Random Forest Classifire for Positive, Neutral, Negative

# In[119]:


train_score = []
test_score = []
F_score = []
Model = [] 

y = Gapp_review.iloc[:,-1:].values
X = text_matrix

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train_std, y_train)
y_pred = rfc.predict(X_test_std)

Model.append('Random Forest Classifier')
train_score.append(rfc.score(X_train_std, y_train))
test_score.append(rfc.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average = 'weighted'))

print('Testing accuracy', rfc.score(X_train_std, y_train))
print('Training accuracy', rfc.score(X_test, y_test))


# In[120]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[121]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(-1, 2):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(-1, 2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
label='macro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["macro"]),
color='green', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(-1, 2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Classifier for Positive(1), Neutral(0), Negative(-1)')
plt.legend(loc="lower right")
plt.show()


# ## KNN for Positive, Neutral, Negative

# In[122]:


y = Gapp_review.iloc[:,-1:].values
X = text_matrix

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

n_neighbors = 5

knn = neighbors.KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test_std)

Model.append('KNN')
train_score.append(knn.score(X_train_std, y_train))
test_score.append(knn.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average = 'weighted'))

print(knn.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[123]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(-1, 2):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(-1, 2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
label='macro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["macro"]),
color='green', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(-1, 2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN for Positive(1), Neutral(0), Negative(-1)')
plt.legend(loc="lower right")
plt.show()


# ## Decision Tree

# In[124]:


y = Gapp_review.iloc[:,-1:].values
X = text_matrix

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test_std)

Model.append('Decision Tree')
train_score.append(dt.score(X_train_std, y_train))
test_score.append(dt.score(X_test_std, y_test))
F_score.append(f1_score(y_test, y_pred, average = 'weighted'))

print("Decision Tree Testing accuracy: ",dt.score(X_test_std,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[125]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(-1, 2):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(-1, 2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
label='macro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["macro"]),
color='green', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(-1, 2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree for Positive(1), Neutral(0), Negative(-1)')
plt.legend(loc="lower right")
plt.show()


# In[126]:


Model_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score', 'F1_score'])

for i in range(3):
    Model_comparison.loc[i, 'Classfier_name'] = Model[i]
    Model_comparison.loc[i, 'train_score'] = train_score[i]
    Model_comparison.loc[i, 'test_score'] = test_score[i]
    Model_comparison.loc[i, 'F1_score'] = F_score[i]
    
Model_comparison


# # App Classification by Positive, Neutral and Negative

# ## Random Forest Classification

# In[117]:


y = Gapp_review.iloc[:,-1:].values
X = app_matrix

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test_std)

print('Testing accuracy', rfc.score(X_train_std, y_train))
print('Training accuracy', rfc.score(X_test_std, y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[104]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(-1, 2):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(-1, 2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
label='macro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["macro"]),
color='green', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(-1, 2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Rand0om Forest Classifier for Positive(1), Neutral(0), Negative(-1)')
plt.legend(loc="lower right")
plt.show()


# ## Decision Tree

# In[114]:


y = Gapp_review.iloc[:,-1:].values
X = app_matrix

class_le = LabelEncoder()
y = class_le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

print("Decision Tree Training accuracy: ",dt.score(X_train,y_train))
print("Decision Tree Testing accuracy: ",dt.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[101]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(-1, 2):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(-1, 2)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(-1, 2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 3
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
label='macro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["macro"]),
color='green', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(-1, 2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
label='ROC curve of class {0} (area = {1:0.2f})'
''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree for Positive(1), Neutral(0), Negative(-1)')
plt.legend(loc="lower right")
plt.show()

