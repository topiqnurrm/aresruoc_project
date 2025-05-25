
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


app_description = pd.read_csv('appleStore_description.csv')
app_description.head()


# In[3]:


app_info = pd.read_csv('AppleStore.csv')
app_info = app_info.drop(columns = ['Unnamed: 0', 'id', 'vpp_lic', 'ver'], axis = 1)
app_info.rename(columns = {'rating_count_ver':'rating_count_cur','user_rating_ver':'user_rating_cur', 'sup_devices.num':'sup_devices_num','ipadSc_urls.num':'screenshot_num', 'lang.num':'lang_num'}, inplace = True)
app_info.head()


# In[4]:


app_description['rating_count_tot'] = app_info['rating_count_tot']
app_description.head()


# ### All Descriptions

# In[34]:


# Find the top three most frequent words
'''
for i in range(app_description.shape[0]):
    R_words = app_description['app_desc'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    R_word_list = tokenizer.tokenize(R_words) 
    # lowercase and remove stopwords
    R_word_list = [word.lower() for word in R_word_list if word not in stopwords.words('english')] 
    # word frequence
    R_word_frequence = nltk.FreqDist(R_word_list)
    sort_freq = pd.Series(R_word_frequence).sort_values(ascending=False)
    if sort_freq.shape[0] >= 3:
        app_description.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
        app_description.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
        app_description.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
    elif sort_freq.shape[0] >= 2:
        app_description.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
        app_description.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    elif sort_freq.shape[0] >= 1:
        app_description.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    else: 
        app_description.loc[i, 'most_freq_word_1st'] = 'NAN'
'''        


# In[5]:


#app_description.to_csv('app_description_most_freq.csv')
app_description_most_freq = pd.read_csv('app_description_most_freq.csv')
app_description_most_freq.head()


# In[27]:


allapp_most_freq1 = app_description_most_freq.most_freq_word_1st.value_counts()
allapp_most_freq1 = allapp_most_freq1.drop('the')
allapp_most_freq1 = allapp_most_freq1.drop('5')
allapp_most_freq1 = allapp_most_freq1.drop('i')
allapp_most_freq1 = allapp_most_freq1.drop('de')
allapp_most_freq1 = allapp_most_freq1.drop('1')
allapp_most_freq1 = allapp_most_freq1.drop('com')
allapp_most_freq1 = allapp_most_freq1.drop('http')
allapp_most_freq1 = allapp_most_freq1.drop('us')


# In[74]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
allapp_most_freq1.head(20).plot.bar()


# In[11]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
app_description_most_freq.most_freq_word_2nd.value_counts().head(20).plot.bar()


# In[12]:


plt.figure(figsize=(16,8))
plt.title('The Third Most Frequent Words')
app_description_most_freq.most_freq_word_3rd.value_counts().head(20).plot.bar()


# #### Word Cloud

# In[13]:


description = app_description_most_freq.app_desc[0]

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(description)

# Display the generated image:
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Top 500 Rating Count

# In[6]:


Rating_count2 = app_description.sort_values('rating_count_tot',ascending = False).head(500)
Rating_count2.reset_index(inplace = True, drop = True)
Rating_count2.head()


# In[7]:


for i in range(Rating_count2.shape[0]):
    R_words = Rating_count2['app_desc'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    R_word_list = tokenizer.tokenize(R_words) 
    # lowercase and remove stopwords
    R_word_list = [word.lower() for word in R_word_list if word not in stopwords.words('english')] 
    # word frequence
    R_word_frequence = nltk.FreqDist(R_word_list)
    sort_freq = pd.Series(R_word_frequence).sort_values(ascending=False)
    Rating_count2.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    Rating_count2.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    Rating_count2.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]


# In[8]:


Rating_count2.head()


# In[47]:


ratecount2_most_freq1 = Rating_count2.most_freq_word_1st.value_counts()
ratecount2_most_freq1 = ratecount2_most_freq1.drop('com')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('5')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('http')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('the')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('99')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('www')
ratecount2_most_freq1 = ratecount2_most_freq1.drop('it')


# In[70]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
ratecount2_most_freq1.head(20).plot.bar()


# In[12]:


plt.figure(figsize=(16,8))
plt.title('The Second Most Frequent Words')
Rating_count2.most_freq_word_2nd.value_counts().head(20).plot.bar()


# In[71]:


ratecount = Rating_count2.app_desc[0]

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(ratecount)

# Display the generated image:
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Top Rating (higher than 4.0)

# In[9]:


app_info['app_desc'] = app_description['app_desc']
app_info.head()


# In[10]:


top_rating = app_info[app_info['user_rating'] >= 4.0]
top_rating.reset_index(inplace = True, drop = True)
top_rating.head()


# In[22]:


# Find the top three most frequent words

for i in range(top_rating.shape[0]):
    words = top_rating['app_desc'][i]
    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+') 
    word_list = tokenizer.tokenize(words) 
    # lowercase and remove stopwords
    word_list = [word.lower() for word in word_list if word not in stopwords.words('english')] 
    # word frequence
    word_frequence = nltk.FreqDist(word_list)
    sort_freq = pd.Series(word_frequence).sort_values(ascending=False)
    top_rating.loc[i, 'most_freq_word_1st'] = sort_freq.index[0]
    top_rating.loc[i, 'most_freq_word_2nd'] = sort_freq.index[1]
    top_rating.loc[i, 'most_freq_word_3rd'] = sort_freq.index[2]
 


# In[11]:


top_rating.to_csv('top_rating.csv')
#top_rating = pd.read_csv('top_rating.csv', index_col=0)
top_rating.head()


# In[59]:


toprate_most_freq1 = top_rating.most_freq_word_1st.value_counts()
toprate_most_freq1 = toprate_most_freq1.drop('the')
toprate_most_freq1 = toprate_most_freq1.drop('com')
toprate_most_freq1 = toprate_most_freq1.drop('http')
toprate_most_freq1 = toprate_most_freq1.drop('5')
toprate_most_freq1 = toprate_most_freq1.drop('i')
toprate_most_freq1 = toprate_most_freq1.drop('us')
toprate_most_freq1 = toprate_most_freq1.drop('a')
toprate_most_freq1 = toprate_most_freq1.drop('you')


# In[72]:


plt.figure(figsize=(16,8))
plt.title('The Most Frequent Words')
toprate_most_freq1.head(20).plot.bar()


# In[33]:


plt.figure(figsize=(12,8))
plt.title('The Second Most Frequent Words')
top_rating.most_freq_word_2nd.value_counts().head(20).plot.bar()


# In[34]:


plt.figure(figsize=(12,8))
plt.title('The Third Most Frequent Words')
top_rating.most_freq_word_3rd.value_counts().head(20).plot.bar()


# In[35]:


#Word cloud
text = app_description.app_desc[0]

# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

