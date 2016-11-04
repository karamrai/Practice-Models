
# coding: utf-8

# In[115]:

#Reading in data and importing necessary packages
import numpy as np
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv'
col_names = ['label', 'message']
sms = pd.read_table(url, sep='\t', header=None, names=col_names)


# sms.head()

# In[7]:

sms.head(20)


# In[8]:

sms.label.value_counts()


# In[11]:

#convert labels into numeric variables
sms['label'] = sms.label.map({'ham':0, 'spam':1})


# In[14]:

#define X and Y
X = sms.message
y = sms.label


# In[15]:

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
print(X_train.shape)
print(X_test.shape)


# sms.head(10)

# In[23]:

sms.head()


# In[93]:

#Vectorizing Data
from sklearn.feature_extraction.text import CountVectorizer


# In[94]:

example_text = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!', 'help']


# In[95]:

#Learning the "vocabulary" of the training data
vect = CountVectorizer()
vect.fit(X_train)
vect.get_feature_names()


# In[96]:

#Creating a document-term matrix with X train data
X_train_dtm = vect.transform(X_train)
X_train_dtm


# In[97]:

#Creating a document-term matrix with X test data
X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[98]:

#Store token names
X_train_tokens = vect.get_feature_names()


# In[99]:

#Show first 50 tokens of training data
print(X_train_tokens[:50])


# In[100]:

#Show last 50 tokens of training data
print(X_train_tokens[-50:])


# In[101]:

#Convert the document term matrix of training data from a sparse matrix to a dense matrix
X_train_dtm.toarray()


# In[102]:

#Count how many times each token appears across ALL messages in X_train_dtm
X_train_counts = np.sum(X_train_dtm.toarray(), axis=0)
X_train_counts


# In[103]:

#Create a dataframe of tokens with their counts
pd.DataFrame({'token':X_train_tokens, 'count':X_train_counts}).sort_values(by='count', ascending=False)


# In[104]:

#Training a Multinomial Naive Bayes model using X_train_dtm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)


# In[105]:

#Make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# In[106]:

#Calculate the accuracy of class predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_class))


# In[107]:

from sklearn.metrics import confusion_matrix


# In[108]:

#Create a confusion matrix
print(confusion_matrix(y_test, y_pred_class))


# In[109]:

#Predict (poorly calibrated) probabilities
y_pred_prob = nb.predict_proba(X_test_dtm)[:1]
y_pred_prob


# In[110]:

#Printing all false positives
X_test[y_test < y_pred_class]


# In[111]:

#Printing all false negatives
X_test[y_test > y_pred_class]


# In[112]:

#Calculate precision and recall
precision = 1202/(1202+4)
recall = 1202/(1202+12)
print(precision)
print(recall)


# In[113]:

#Calculate f-measure
f_measure = 2*((precision*recall)/(precision+recall))
print(f_measure)

