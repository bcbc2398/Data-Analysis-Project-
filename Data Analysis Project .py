#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


dataset = pd.read_csv('verified_nft_tweets.csv')


# In[29]:


dataset


# In[30]:


from textblob import TextBlob

def clean(tweet):
    return str(tweet).encode('ascii', 'ignore').decode('UTF-8')

def get_subjectivity(tweet):
    return round(TextBlob(tweet).sentiment.subjectivity, 2)

def get_polarity(tweet):
    return round(TextBlob(tweet).sentiment.polarity, 2)

dataset['tweet'] = dataset['tweet'].apply(clean)
dataset['subjectivity'] = dataset['tweet'].apply(get_subjectivity)
dataset['polarity'] = dataset['tweet'].apply(get_polarity)
dataset.drop(dataset[dataset['polarity'] == 0].index, inplace=True)
dataset.drop_duplicates(subset='tweet', keep="first", inplace=True)


# In[31]:


dataset


# In[32]:


import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
sid = SentimentIntensityAnalyzer()


# In[33]:


dataset['scores'] = dataset['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))


# In[34]:


dataset


# In[35]:


dataset['compound'] = dataset['scores'].apply(lambda score_dict: score_dict['compound'])
dataset['sentiment_type']=''
dataset.loc[dataset.compound > 0, 'sentiment_type']='POSITIVE'
dataset.loc[dataset.compound == 0, 'sentiment_type']='NEUTRAL'
dataset.loc[dataset.compound < 0, 'sentiment_type']='NEGATIVE'

from textblob import TextBlob

def clean(tweet):
    return str(tweet).encode('ascii', 'ignore').decode('UTF-8')

def get_subjectivity(tweet):
    return round(TextBlob(tweet).sentiment.subjectivity, 2)

def get_polarity(tweet):
    return round(TextBlob(tweet).sentiment.polarity, 2)

dataset['tweet'] = dataset['tweet'].apply(clean)
dataset['subjectivity'] = dataset['tweet'].apply(get_subjectivity)
dataset['polarity'] = dataset['tweet'].apply(get_polarity)
dataset.drop(dataset[dataset['polarity'] == 0].index, inplace=True)
dataset.drop_duplicates(subset='tweet', keep="first", inplace=True)


# In[36]:


dataset


# In[37]:


dataset.drop(['cashtags','quoted_user','video','reply_to_user',],axis=1,inplace=True)


# In[38]:


dataset


# In[39]:


dataset['polarity'].hist(edgecolor='black', color='green');
plt.title('Polarity of tweets')
plt.xlabel('Polarity')
plt.ylabel('Number of tweets')


# In[40]:


dataset['subjectivity'].hist(edgecolor='black', color='yellow');
plt.title('Subjectivity of tweets')
plt.xlabel('Subjectivity')
plt.ylabel('Number of tweets')


# In[41]:


plt.figure(figsize=(12,8))
plt.scatter(dataset['subjectivity'], dataset['polarity'], alpha=1/3, s=3, color='Blue')
plt.title('Sentiment Analysis', size=20)
plt.xlabel('Subjectivity', size=18)
plt.ylabel('Polarity', size=18)


# In[42]:


dataset.plot(kind='scatter',x='polarity',y='likes_count',alpha=1/3, color='red')
plt.title('Sentiment Analysis')


# In[43]:


dataset.groupby('polarity')['subjectivity'].value_counts().sort_index().plot()
plt.xlabel('Polarity/Subjectivity', size=18)
plt.ylabel('Tweets', size=18)
plt.title('Sentiment Analysis', size=20, color='blue', loc='left')


# In[44]:


plt.figure(figsize=(12,8))
plt.scatter(dataset['subjectivity'], dataset['likes_count'], alpha=1/3, s=3, color='Blue')
plt.title('Sentiment Analysis', size=20)
plt.xlabel('Subjectivity', size=18)
plt.ylabel('Likes_count', size=18)


# In[45]:


dataset['username'].value_counts().sort_index().head(50).plot(kind='bar', figsize=(20,10))
plt.xlabel('username', size=18)
plt.ylabel('tweet count', size=18)
plt.title('Top 50 Most Frequent Tweeters', size=20, color='blue',)


# In[46]:


sns.set_style('darkgrid')


# In[47]:


sns.countplot(x='polarity', data=dataset.head(20));plt.title('Sentiment')


# In[48]:


dataset.groupby(['username','tweet'])['compound'].mean().sort_values(ascending=True).head(60)


# In[49]:


dataset.groupby(['username','tweet'])['compound'].mean().sort_values(ascending=False).head(60)


# In[50]:


dataset.groupby(['username','tweet'])['polarity'].mean().sort_values(ascending=True).head(60)


# In[51]:


dataset.groupby(['username','tweet'])['polarity'].mean().sort_values(ascending=False).head(60)


# In[52]:


# Lowest compound 
davidbianchi     My @BoredApeYC guy is officially worth $36,260 
#NFTs are fucking DEAD  #bayc #BoredApeYachtClub #nft #nftcollector   DEAD. FUCKING. DEAD.  https://t.co/0KBiKYMsB3                                                                                                                                                     -0.9522


# In[ ]:


#Highest Polarity
jordinsparks     Yeeee proud owner of this lil #NFT Baby Ape #29! Say hiiii! @BabyApe_SC #29  https://t.co/gVsIxUkjER                                                                                                                                                                                                                 1.0


# In[ ]:


# Highest compound 
callmelatasha    The selling is amazing but what has been on my mind is breaking ground for diverse mediums winning in #nft.
I love the 3D works I see win but I am excited to see culture win. People win. Poetry win. Performance win. Photography win. Film win. 


# In[ ]:


#Lowest Polarity
glynmoody       quite; outrageous that media are pushing out this garbage #NFT    -1.0


# In[53]:


df = dataset.groupby(['username'])['polarity'].mean().sort_values(ascending=False).head(20)
plt.figure(figsize=(10,4))
plt.bar(df.index, df, color='blue')
plt.xlabel('Sentiment')
plt.ylabel('Polarity')
plt.xticks(rotation=90)
plt.title("Top 20 most polar positive tweets", size=12)


# In[54]:


df = dataset.groupby(['username'])['polarity'].mean().sort_values(ascending=True).head(20)
plt.figure(figsize=(10,4))
plt.bar(df.index, df, color='blue')
plt.xlabel('Sentiment')
plt.ylabel('Polarity')
plt.xticks(rotation=90)
plt.title("Top 20 most polar negative tweets", size=12)


# 

# In[55]:


plt.figure(figsize=(12,8))
plt.scatter(dataset['polarity'], dataset['likes_count'], alpha=1/3, s=3, color='Blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Likes_count')


# In[60]:


df_plot = dataset.groupby(['polarity'])['likes_count'].agg(['mean']).plot()
plt.ylabel('Avg. retweets count')
plt.title('Polarity of Avg. retweets count')


# In[ ]:






# In[ ]:





# In[ ]:




