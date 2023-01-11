#Here are what the columns represent:
#credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
#purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
#int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
#installment: The monthly installments owed by the borrower if the loan is funded.
#log.annual.inc: The natural log of the self-reported annual income of the borrower.
#dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
#fico: The FICO credit score of the borrower.
#days.with.cr.line: The number of days the borrower has had a credit line.
#revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
#revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
#inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
#delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
#pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Get the Data
#Use pandas to read loan_data.csv as a dataframe called loans.**
loans = pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()


#Exploratory Data Analysis
#Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

#Create a similar figure, except this time select by the not.fully.paid column.**
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

#Let's see the trend between FICO score and interest rate.
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

#Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy.
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

#Setting up the Data
#Let's get ready to set up our data for our Random Forest Classification Model!
#Check loans.info() again.**
loans.info()

#Categorical Features
#Notice that the **purpose** column as categorical
#That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
#Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
#Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
cat_feats = ['purpose']

#Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()

#Train Test Split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

#Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


#Predictions and Evaluation of Decision Tree
#Create predictions from the test set and create a classification report and a confusion matrix.
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#Training the Random Forest model
#Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

#Predictions and Evaluation
#Let's predict off the y_test values and evaluate our model.
#Predict the class of not.fully.paid for the X_test data.**
predictions = rfc.predict(X_test)

#Now create a classification report from the results. Do you get anything strange or some sort of warning?
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

#Show the Confusion Matrix for the predictions.**
print(confusion_matrix(y_test,predictions))





