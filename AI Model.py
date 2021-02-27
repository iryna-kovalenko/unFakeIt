import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
model = joblib.load(open("/home/ubuntu/Model_Testing/LogisticRegression.pkl", 'rb'))
read_test=pd.read_csv('/home/ubuntu/Model_Testing/test.csv')
tf1= pickle.load(open("/home/ubuntu/Model_Testing/vectorizer.pkl","rb"))
col=read_test['text']
empt_list_test=[] 
for l in col: 
  empt_list_test.append(l)
vectorizer=CountVectorizer(vocabulary=tf1.vocabulary_)
vectorized_sentence_1=vectorizer.transform(empt_list_test[20:100]) 
print(model.predict(vectorized_sentence_1))