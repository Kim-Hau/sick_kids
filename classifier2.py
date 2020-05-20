import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from random import randint

df = pd.read_csv("2018_02_03_rdtw.csv", names = ["timestamp", "DTW"])
windows_length = 1000
positive_windows = pd.DataFrame()
negative_windows = pd.DataFrame()

#positive_windows = positive_windows.append(pd.Series(list(df.iloc[0:10,1])), ignore_index = True)
#negative_windows = negative_windows.append(pd.Series(list(df.iloc[1:11,1])), ignore_index = True)

classification = list()
p_class = list()
n_class = list()
p_dtw = list()
p_ts = list()
n_dtw = list()
n_ts = list()
start_classif = list()
end_classif = list()

for i in range(0,len(df.timestamp)-windows_length,100):
    end = i+windows_length-1
    if 154404-windows_length<= end <= 157154 or 1194363-windows_length <= end <= 1197488 or 1622530-windows_length <= end <= 1625655 or 1632155-windows_length <= end <= 1634155 or 1706280-windows_length <= end <= 1709530 or 3142281-windows_length <= end <= 3144906 or 6099075-windows_length <= end <= 6101825 or 6208700-windows_length <= end <= 6211325 or 6605242-windows_length <= end <= 6606742 or 6648492-windows_length <= end <= 6651117 or 8712702-windows_length <= end <= 8717202:
        classification.append(1)
        p_class.append(1)
        positive_windows = positive_windows.append(pd.Series(list(df.iloc[i:windows_length+i,1])), ignore_index = True)
        p_dtw.append(df.DTW[end])
        p_ts.append(df.timestamp[end])
        if classification[len(classification)-2] == 0:
            start_classif.append(df.timestamp[i-1])
    else:
        n_class.append(0)
        classification.append(0)
        n_dtw.append(df.DTW[end])
        n_ts.append(df.timestamp[end])
        negative_windows = negative_windows.append(pd.Series(list(df.iloc[i:windows_length+i,1])), ignore_index = True)
        if classification[len(classification)-2] == 1:
            end_classif.append(df.timestamp[i-1])

print("done")
#positive_windows["class"] = (p_class)
#negative_windows["class"] = (n_class)
print(negative_windows)

gt_timestamps = pd.DataFrame()
gt_timestamps["start"] = start_classif
gt_timestamps["end"] = end_classif
gt_timestamps.to_csv(str(windows_length)+"timestamps.csv", index = False, header = False)
positive_windows.to_csv(str(windows_length)+"positive_windows.csv", index = False, header = False)
negative_windows.to_csv(str(windows_length)+"negative_windows.csv", index = False, header= False)

test_windows = [positive_windows, negative_windows]
test_windows = pd.concat(test_windows)

add_windows = len(p_class)
count = 0
added_indices = list()
test_class = p_class[:]
while count<= (int(add_windows/40*60)):
    rand = randint(0, len(n_class))
    if not (rand in added_indices):
        positive_windows = positive_windows.append(negative_windows.iloc[rand], ignore_index = True)
        added_indices.append(rand)
        count+=1
        test_class.append(0)

print("done2")
x_train = positive_windows
y_train = test_class
#y_train=y_train.astype('int')

x_test= test_windows
p_class+= n_class
y_test = p_class
#y_test=y_test.astype('int')

logistic_regression = LogisticRegression(max_iter=15)
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100*accuracy
print(accuracy_percentage)

predictions = dict()
p_ts+=(n_ts)
p_dtw+=(n_dtw)
predictions["ts"] = pd.Series(p_ts)
predictions["val"] = pd.Series(y_pred)
predictions["dtw"] = pd.Series(p_dtw)

predictions = pd.DataFrame(predictions)
predictions.to_csv(str(windows_length)+"predictions.csv", index = False, header = False)