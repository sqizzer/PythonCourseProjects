import seaborn as sns
iris = sns.load_dataset('iris')
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Create a pairplot of the data set.
sns.pairplot(iris,hue='species',palette='Dark2')

#Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)

#Train Test Split
#Split your data into a training set and a testing set.**
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Train a Model
#Now its time to train a Support Vector Machine Classifier. 
#Call the SVC() model from sklearn and fit the model to the training data.

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

#Model Evaluation
#Now get predictions from the model and create a confusion matrix and a classification report.
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Import GridsearchCV from SciKit Learn.
from sklearn.model_selection import GridSearchCV

#Create a dictionary called param_grid and fill out some parameters for C and gamma.
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

#Create a GridSearchCV object and fit it to the training data.**
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

#Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them.
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


