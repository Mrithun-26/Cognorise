import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
# print(df.head())
data = df.drop(columns=['date', 'street', 'city', 'statezip', 'country'])
df['date'] = pd.to_datetime(df['date'])
# print(df.tail(5))
col = []
for i, v in enumerate(df.columns, 1):
    col.append({i, v})
print(col)
df.info()
print(df.index)
df.describe()
df.isnull().sum()
print(f"duplicated number --> {df.duplicated().sum()}")
print(f" the highest price {df['price'].max()}")  # Highest house price
print(f" Lowest price {df['price'][df.price > 0].min()}")  # Lowest house price
# Represent distribution Price on a specific day
b = np.arange(stop=2384000, start=235000, step=100000)
sns.histplot(df, x=df['price'][df['date'] == '2014-05-02 00:00:00'], kde=True, bins=b)
plt.title('distribution of price in 2014-05-02')
sns.scatterplot(data=df, x='sqft_living', y='price')  # Most sqft_living are in between 8000-2000
l = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view']
for i, v in enumerate(l, 1):
    plt.figure(figsize=(3, 3))
    plt.subplot(6, 1, i)
    sns.lmplot(df, x=v, y='price')
# plt.show()
plt.figure(figsize=(20, 10))
sns.countplot(df.head(50), x='city')
plt.title('distribution for first 50 row ')
l = ['bedrooms', 'bathrooms', 'floors']
plt.figure(figsize=(10, 10))
for i, v in enumerate(l, 1):
    plt.subplot(3, 1, i)
    sns.histplot(df[v], kde=True, palette='plasma')
    plt.title(f'Distribution for {v}')
    plt.xlabel(v)
    plt.ylabel('Frequency')
plt.tight_layout()
sns.countplot(x="bedrooms", data=df, width=0.1)
plt.title('distribution of bedrooms  ')
# plt.show()
sns.pairplot(df)
# correlation matrix
corr_matrix = data.corr(method="pearson")
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
high_corr_feature = corr_matrix.index[abs(corr_matrix['price']) >= 0.2].tolist()
high_corr_feature.remove('price')  # Remove the target variable from the list
# print(corr_matrix)
print(high_corr_feature)
# model
y = df['price']
x = df[high_corr_feature]
scaler = StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.012552, random_state=58)
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
print(f"x train examples -->{x_train.shape[0]} from {x.shape[0]}")
print(f"x test examples -->{x_test.shape[0]} from {y.shape[0]}")
reg = LinearRegression()
reg.fit(x_train, y_train)
y_test_pre = reg.predict(x_test)
test_acc = r2_score(y_test, y_test_pre)
print(test_acc)
