# imoprt modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import folium as fm


data = pd.read_csv(r"covid_19_clean_complete_2022.csv")

print(data.shape)
print(data.columns.values)
print(data.isnull().sum())

data = data.dropna()
print(data.isnull().sum())
print(data.dtypes)

op = ["Province/State","Country/Region","WHO Region"]

for x in op:
    La = LabelEncoder()
    data[x] = La.fit_transform(data[x])
    
print(data.dtypes)

print(data["Date"].head())
data["Date"] = pd.to_datetime(data["Date"])
data["day"] = data["Date"].dt.day
data["month"] = data["Date"].dt.month
data["year"] = data["Date"].dt.year
data = data.drop("Date", axis=1)
print(data.dtypes)

plt.figure(figsize=(14,7))
sns.heatmap(data.corr(), annot=True)
plt.show()

data["died"] = 0
data.loc[data["Deaths"] > 0, "died"] = 1
print(data["died"].value_counts())


x = data.drop("died", axis=1)
y = data["died"]

ss = StandardScaler()
x = ss.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle =True)
print(X_train.shape)


Lo = LogisticRegression()
Lo.fit(X_train, y_train)

print("_"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))
print("_"*100)


# print("_"*150)
# for x in range(2,20):
#     Dt = DecisionTreeClassifier(max_depth=x,random_state=33)
#     Dt.fit(X_train, y_train)

#     print("x = ", x)
#     print(Dt.score(X_train, y_train))
#     print(Dt.score(X_test, y_test))
#     print("_"*100)




Dt = DecisionTreeClassifier(max_depth=3,random_state=33)
Dt.fit(X_train, y_train)

print("_"*100)
print(Dt.score(X_train, y_train))
print(Dt.score(X_test, y_test))
print("_"*100)
y_pred = Dt.predict(X_test)

# confusion_matrix
Cm = confusion_matrix(y_test,y_pred)
print(Cm)
sns.heatmap(Cm,annot=True, fmt="d", cmap="hot")
plt.show()

# accuracy_score
print("_"*100)
As = accuracy_score(y_test,y_pred)
print(As)

# The autput result
result = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# result.to_csv("The autput.csv",index=False)

# df = data[:1000]
# m = fm.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)

# for x,y in zip(df["Lat"],df["Long"]):
#     fm.Marker([x,y]).add_to(m)
    
    

