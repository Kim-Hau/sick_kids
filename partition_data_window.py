import pandas as pd

df = pd.read_csv("2018_02_03_o0_l3751_h100.csv", names = ["timestamp", "DTW Distance"])
#863659 values

l = list()
l2 = list()
d = dict()
length = 100

for i in range(length):
	l=list( df.iloc[i:i+863659-length,1])
	d[i] = pd.Series(l)

for i in range (863659-length):
	if (1517635237000<=df["timestamp"].values[i]<(1517635259000-100) or 1517643560000<=df["timestamp"].values[i]<(1517643585000-100) or 1517646987000<=df["timestamp"].values[i]<(1517647012000-100) or 1517647064000<=df["timestamp"].values[i]<(1517647080000-100) or 1517647657000<=df["timestamp"].values[i]<(1517647683000-100) or 1517659150000<=df["timestamp"].values[i]<(1517659171000-100) or 1517682815000<=df["timestamp"].values[i]<(1517682837000-100) or 1517683692000<=df["timestamp"].values[i]<(1517683713000-100) or 1517686866000<=df["timestamp"].values[i]<(1517686878000-100) or 1517687212000<=df["timestamp"].values[i]<(1517687233000-100) or 1517703734000<=df["timestamp"].values[i]<(1517703770000-100)):
		l2.append(1)
	else:
		l2.append(0)

d["Classification"] = pd.Series(l2)

df2 = pd.DataFrame(d)
df2.to_csv("windows.csv", index = False)
print(df2)

