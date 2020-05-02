import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('partition.csv')
df.head()
x = df.drop('Classification', axis=1)
y = df.Classification

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

logistic_regression = LogisticRegression()

logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100*accuracy
print(accuracy_percentage)