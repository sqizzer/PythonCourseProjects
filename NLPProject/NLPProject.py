import numpy as np
import pandas as pd

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()


#Create a new column called "text length" which is the number of words in the text column.
yelp['text length'] = yelp['text'].apply(len)

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')

#Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')

#Create a boxplot of text length for each star category.**
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')

#Create a countplot of the number of occurrences for each type of star rating.
sns.countplot(x='stars',data=yelp,palette='rainbow')

#Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:
stars = yelp.groupby('stars').mean()
stars

#Use the corr() method on that groupby dataframe to produce this dataframe:
stars.corr()

#Then use seaborn to create a heatmap based off that .corr() dataframe:
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

#NLP Classification Task
#Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

#Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)
X = yelp_class['text']
y = yelp_class['stars']

#Import CountVectorizer and create a CountVectorizer object.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.
X = cv.fit_transform(X)

#Train Test Split
#Let's split our data into training and testing data.
#Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


#Training a Model
#Import MultinomialNB and create an instance of the estimator and call is nb
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#Now fit nb using the training data.**
nb.fit(X_train,y_train)

#Predictions and Evaluations
#Use the predict method off of nb to predict labels from X_test.**
predictions = nb.predict(X_test)

#Create a confusion matrix and classification report using these predictions and y_test 
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

#Import TfidfTransformer from sklearn.
from sklearn.feature_extraction.text import  TfidfTransformer

#Import Pipeline from sklearn. 
from sklearn.pipeline import Pipeline

#Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
#Using the Pipeline
#Train Test Split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

#Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels
pipeline.fit(X_train,y_train)

#Predictions and Evaluation
#Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



