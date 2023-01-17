import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('lending_club_loan_two.csv')
df.info()

sns.countplot(data=df,x='loan_status')
plt.figure(figsize=(12,6))

sns.histplot(data=df,x='loan_amnt')
plt.figure(figsize=(12,6))

sns.distplot(df['loan_amnt'],kde=False)
df.corr()

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='magma')
df['installment'].describe()
plt.figure(figsize=(12,6))

sns.scatterplot(data=df,x='loan_amnt',y='installment')

sns.boxplot(data=df,x='loan_status',y='loan_amnt')
df.groupby('loan_status')['loan_amnt'].describe()
df.columns

np.sort(df['grade'].unique())
np.sort(df['sub_grade'].unique())

sns.countplot(data=df,x='grade',hue='loan_status')
plt.figure(figsize=(14,7))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(data=df,x='sub_grade',order=subgrade_order)

plt.figure(figsize=(14,7))
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(data=f_and_g,x='sub_grade',order=subgrade_order)


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

df.head()

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

len(df)
100*df.isnull().sum()/len(df)


df['emp_title'].nunique()
df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)
sorted(df['emp_length'].dropna().unique())

emp_length_order = ['1 year',
 '< 1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years']

plt.figure(figsize=(12,6))
sns.countplot(data=df,x='emp_length',order=emp_length_order)

plt.figure(figsize=(12,6))
sns.countplot(data=df,x='emp_length',order=emp_length_order,hue='loan_status')


emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']
(emp_co/(emp_fp + emp_co)).plot(kind='bar')

df = df.drop('emp_length',axis=1)
df.isnull().sum()

df['purpose'].head(10)
df['title'].head(10)

df = df.drop('title',axis=1)

df['mort_acc'].describe()
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc,mort_acc):
    
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)
df.isnull().sum()
df = df.dropna()
df.isnull().sum()

df.select_dtypes(['object']).columns

df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))
df['term'].value_counts()

df = df.drop('grade',axis=1)
dummies = pd.get_dummies(df['sub_grade'],drop_first = True)

df = pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)

df.columns

dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first = True)

df = pd.concat([df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1),dummies],axis=1)

df.columns

df['home_ownership'].value_counts()
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)
df.columns

df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df['zip_code']

dummies = pd.get_dummies(df['zip_code'],drop_first=True)

df = pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)

df.columns

df = df.drop('address',axis=1)
df = df.drop('issue_d',axis=1)

df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df['earliest_cr_line'].value_counts()

from sklearn.model_selection import train_test_split

df = df.drop('loan_status',axis=1)
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

print(len(df))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

X_train.shape


# In[100]:


model = Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units = 1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,
         validation_data=(X_test,y_test))

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import classification_report,confusion_matrix
predictions = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer

new_customer = scaler.transform(new_customer.values.reshape(1,78))
model.predict_classes(new_customer)


# In[ ]:




