import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from random import randint


false_positives = 0
false_negatives = 0
true_pos = 0
true_neg = 0
count = 0


df = pd.read_csv("2018_02_03_o0_l3751_h100.csv", names = ["timestamp", "DTW Distance"])
#863659 values

col = list()
class = list()
data = dict()
length = 1000

for i in range(length):
	col=list( df.iloc[i:i+863659-length,1])
	data[i] = pd.Series(col)

for i in range (length-1, 863659-1):
	if (1517635237000+length*4<df["timestamp"].values[i]<=(1517635259000) or 1517643560000+length*4<df["timestamp"].values[i]<=(1517643585000) or 1517646987000+length*4<df["timestamp"].values[i]<=(1517647012000) or 1517647064000+length*4<df["timestamp"].values[i]<=(1517647080000) or 1517647657000+length*4<df["timestamp"].values[i]<=(1517647683000) or 1517659150000+length*4<df["timestamp"].values[i]<=(1517659171000) or 1517682815000+length*4<df["timestamp"].values[i]<=(1517682837000) or 1517683692000+length*4<df["timestamp"].values[i]<=(1517683713000) or 1517686866000+length*4<df["timestamp"].values[i]<=(1517686878000) or 1517687212000+length*4<df["timestamp"].values[i]<=(1517687233000) or 1517703734000+length*4<df["timestamp"].values[i]<=(1517703770000)):
		class.append(1)
	else:
		class.append(0)

data["Classification"] = pd.Series(class)

df = pd.DataFrame(data)
#df2.to_csv("windows.csv", index = False)

added_indices = list()
ones_count = 0
df2 = pd.DataFrame(columns = df.columns)
for i in range (len(df.Classification)):
	if df.at[i, "Classification"]==1:
		df2 = df2.append(df.iloc[i],ignore_index=True)
		added_indices.append(i)
		ones_count+=1


while (count <=int(ones_count/40*60)):
	value = randint(0, len(df.Classification))
	if not (value in added_indices):
		df2 = df2.append(df.iloc[value],ignore_index=True)
		added_indices.append(value)
		count+=1

print(len(df2.Classification))
df.head()
df2.head()
x_train = df2.drop('Classification', axis=1)
y_train = df2.Classification
y_train=y_train.astype('int')

x_test = df.drop('Classification', axis=1)
y_test = df.Classification
y_test=y_test.astype('int')


logistic_regression = LogisticRegression(max_iter=15)
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100*accuracy
print(accuracy_percentage)
test_y = list(y_test)
pred_y = list(y_pred)


for i in range (len(pred_y)):
	if pred_y[i] == 1 and test_y[i] == 0:
		false_positives+=1
	if pred_y[i] == 0 and test_y[i] == 1:
		false_negatives+=1
	if pred_y[i] == 0 and test_y[i] == 0:
		true_neg+=1
	if pred_y[i] == 1 and test_y[i] == 1:
		true_pos+=1


print ("False +: "+str(false_positives))
print ("False -: "+str(false_negatives))
print ("True +: "+str(true_pos))
print ("True -: "+str(true_neg))
