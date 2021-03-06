#Labels are binary 1 is real and 0 is Fake 

#practicing regilar expressions 
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------Using cropped data-------------------------------
basiccsv=pd.read_csv('fake_or_real_news.csv')
data_split=np.random.rand(len(basiccsv))<0.50# take true/false table to split the data, all true will go traiting , all false will fo testing data
df_with_links=basiccsv[~data_split]
print(df_with_links)
#---------------------Using cropped data-------------------------------

#df_testing=pd.read_csv('capstone_file.csv')
print('first')
print(df_with_links['label'].value_counts())

#PREPROCESSING 
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
# method to remove URL's, @ and # from the text
def remove_urls_at_hash(new_list,data_column):
    for t in data_column:
        newT=re.sub('http://\S+|https://\S+','',t) #removing http/https
        no_at=re.sub('@\S+|#\S+|\n|RT|rt','',newT) # removing @,#,\n,rt
        if newT!=no_at : 
            newT=no_at
        if t==newT:
            new_list.append(t)
        elif t!=newT:
            new_list.append(newT)
        
 
#method to remove 1)punctuation 2)stop_words           
def text_cleaning(txt):
    cleaned_tweets=[] #list
    def hasNumbers(inputString):
      return any(char.isdigit() for char in inputString)
    no_dig_tweets=[]  
    temp_no_dig_tweets=[] 
    for tw_1 in txt:
      for this_word in tw_1.split():
        if not hasNumbers(this_word):
          temp_no_dig_tweets.append(this_word.lower())
        word=" ".join(temp_no_dig_tweets)
      no_dig_tweets.append(word)
      temp_no_dig_tweets=[]

    for tweet in no_dig_tweets:
        text_remove_punt=[char for char in tweet if char not in string.punctuation] #in char is not in punctuation list then return
        text_gather_chars_into_words=''.join(text_remove_punt).split()#make 1 long string and split into words
        text_remove_stop_words=[w for w in text_gather_chars_into_words if w.lower() not in stop_words] #if word not in list then  return 
        cleaned_tweets.append(text_remove_stop_words)
    return cleaned_tweets

# method to remove duplicate tweets 
def remove_duplicates(df_for_removing_duplicates):
    no_dub=pd.DataFrame.drop_duplicates(df_for_removing_duplicates)
    return no_dub

#METHOD CALLS
no_duplicates=remove_duplicates(df_with_links)
print('no dublicates first one ')
print(no_duplicates['label'].value_counts())

rem=[]   #list
remove_urls_at_hash(new_list=rem,data_column=no_duplicates['text'])
clean_data=text_cleaning(rem) # removing punctuation, stop words

#to make a from a nested list regular list
clean_data_list=[]
for ii in clean_data:
    clean_data_list.append(' '.join(ii))
print('len of the clean data after all pre processing is :')
print(len(clean_data_list))

final_clean_df=pd.DataFrame(clean_data_list) # casting into Data Frame, so we can use it for vectorization
final_clean_set=list(final_clean_df[0])

#Beg of Words using CountVectorizer from sklearn


vectorizer=CountVectorizer()
x=vectorizer.fit_transform(final_clean_set)#get vocab then build vector then make an array
features_extracted=vectorizer.vocabulary_


# i need list [] --> pu labels in the list 
future_label=[]
for r in no_duplicates['label']:
  future_label.append(r)
print(future_label)
# NEXT STEP IS TO ADD LABELING TO OUR DATASET
# 1.1 after vectorization we have as a result a matrix
matrix_to_array=x.toarray() #1.2 make matrix to array
array_to_df=pd.DataFrame(matrix_to_array)# an array to dataframe
array_to_df['label']=future_label # now append label to the df
print(array_to_df.shape)
print(array_to_df['label'].value_counts())
#print(df_no_duplicate)
#print(rem)
#use collection import counter to see what words are mostly used to identofi stopwords

print('finally  ')
print(array_to_df['label'].value_counts())
#--------------------------------vectorizing clean testing data----------------------------------

#"""Splitting the data"""

# DEVIDE INTO TRAINING AND TESTING SET 
data_split=np.random.rand(len(array_to_df))<0.80# take true/false table to split the data, all true will go traiting , all false will fo testing data
train=array_to_df[data_split] # training set based on randomly selested values 80%
test=array_to_df[~data_split] # testing set based on 20% randomly seleted values
print(' Size of training set : ', len(train))
print(' Size of testing set : ', len(test))
#print(features_extracted) # gives the vocablulary list
print('training data shape is')
print(x.shape) # representation of (tweet number, voablulary list)

# split into independednt and dependednt value for training
leng=len(train.axes[1])-1
x_train=train.iloc[:,0:leng]
y_train=train.iloc[:,leng]

# split into independednt and dependednt value for testing
x_test=test.iloc[:,0:leng]
y_test=test.iloc[:,leng]

print(array_to_df['label'].value_counts())


from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.5, solver='saga',multi_class='multinomial',class_weight='balanced',max_iter=200).fit(x_train,y_train)
yhat=LR.predict(x_test)
print('metrics.accuracy_score : ',metrics.accuracy_score(y_test,yhat ))
