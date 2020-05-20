import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from random import randint

df = pd.read_csv("2018_02_03_rdtw.csv", names = ["timestamp", "DTW"])
gt_values = pd.read_csv("gt_values.csv")

windows_length = 100
windows = dict()

windows["end_timestamp"] = pd.Series(list(df.iloc[windows_length-1:len(df.timestamp)-1,0]))
for i in range (windows_length):
    col=list( df.iloc[i:i+len(df.timestamp)-windows_length,1])
    windows[i] = pd.Series(col)

start_classif = list()
end_classif = list()
classification = list()
flag = 1

for i in range (windows_length-1, len(df.timestamp)-1):
    if 154404-windows_length<= i <= 157154 or 1194363-windows_length <= i <= 1197488 or 1622530-windows_length <= i <= 1625655 or 1632155-windows_length <= i <= 1634155 or 1706280-windows_length <= i <= 1709530 or 3142281-windows_length <= i <= 3144906 or 6099075-windows_length <= i <= 6101825 or 6208700-windows_length <= i <= 6211325 or 6605242-windows_length <= i <= 6606742 or 6648492-windows_length <= i <= 6651117 or 8712702-windows_length <= i <= 8717202:
        
	  classification.append(1)
        if classification[len(classification)-2] == 0:
            start_classif.append(df.timestamp[i-1])
    else:
        classification.append(0)
        if classification[len(classification)-2] == 1:
            end_classif.append(df.timestamp[i-1])
    """
    for j in range(len(gt_values.start)):
        if gt_values.at[j, "start_index"]-windows_length <= i <= gt_values.at[j, "end_index"]:
            classification.append(1)
            if classification[len(classification)-2] == 0:
                start_classif.append(df.timestamp[i-1])
            flag = 0
            break
    if flag == 1:
        classification.append(0)
        if classification[len(classification)-2] == 1:
            end_classif.append(df.timestamp[i-1])
    flag = 1
    """
print("done")
print(len(classification))
print(len(windows["end_timestamp"]))
print(len(col))

windows["classification"] = pd.Series(classification)
classif_intervals = dict()
classif_intervals["start_timestamp"] = pd.Series(start_classif)
classif_intervals["end_timestamp"] = pd.Series(end_classif)

classif_intervals = pd.DataFrame(classif_intervals)
windows = pd.DataFrame(windows)

classif_intervals.to_csv(str(windows_length)+"classification_timestamps.csv", index = False, header = False)
windows.to_csv(str(windows_length)+"windows.csv", index = False)

added_indices = list()
ones_count = 0
df2 = pd.DataFrame(columns = windows.columns)
for i in range (len(windows.classification)):
    if windows.at[i, "classification"] ==1:
        df2 = df2.append(windows.iloc[i],ignore_index=True)
        added_indices.append(i)
        ones_count+=1
print(ones_count)
count = 0
while (count <=int(ones_count/40*60)):
    value = randint(0, len(windows.classification))
    if not (value in added_indices):
        df2 = df2.append(windows.iloc[value],ignore_index=True)
        added_indices.append(value)
        count+=1
print("done2")
windows.head()
df2.head()

x_train = df2.drop('classification', axis=1).drop('end_timestamp', axis=1)
y_train = df2.classification
y_train=y_train.astype('int')

x_test = windows.drop('classification', axis=1).drop('end_timestamp', axis=1)
y_test = windows.classification
y_test=y_test.astype('int')


logistic_regression = LogisticRegression(max_iter=15)
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100*accuracy
print(accuracy_percentage)
test_y = list(y_test)
pred_y = list(y_pred)

predictions = dict()
predictions["ts"] = windows.end_timestamp
predictions["val"] = pd.Series(pred_y)
predictions["dtw"] = windows[length-1]

predictions = pd.DataFrame(predictions)
predictions.to_csv(str(length)+"predictions.csv", index = False, header = False)
