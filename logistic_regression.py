import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from random import randint
import matplotlib.pyplot as plt

df = pd.read_csv('windows.csv')
#2459 instances of 1
#3689 instances of 0 for a 40%-60% ratio
l = list()
one_occurrences = 0
df2 = pd.DataFrame(columns = df.columns)
for i in range (len(df.Classification)):
	if df.at[i, "Classification"]==1:
		df2 = df2.append(df.iloc[i],ignore_index=True)
		l.append(i)
		one_occurrences +=1

count = 0

while (count < int(one_occurrences/40*60)):
	value = randint(0, len(df.Classification))
	if not (value in l):
		df2 = df2.append(df.iloc[value],ignore_index=True)
		l.append(value)
		count+=1

df2.head()
x = df2.drop('Classification', axis=1)
y = df2.Classification
y=y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=4)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

accuracyscore = list()
number_iter = list()
for i in range (1,1000,5):
	logistic_regression = LogisticRegression(max_iter=i)
	logistic_regression.fit(x_train, y_train)

	y_pred = logistic_regression.predict(x_test)
	
	accuracy = metrics.accuracy_score(y_test, y_pred)
	accuracy_percentage = 100*accuracy
	accuracyscore.append(accuracy_percentage)
	number_iter.append(i)


plt.plot(number_iter, accuracyscore) 
plt.xlabel('Number of iterations') 
plt.ylabel('accuracy_score')
plt.show() 