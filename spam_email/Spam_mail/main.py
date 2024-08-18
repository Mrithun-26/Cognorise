import numpy as np  # linear algebra
import pandas as pd  # data processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('spam.csv')
df.info()
df.duplicated()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
df_new = ohe.fit_transform(df[['Category']])
df_new = pd.DataFrame(df_new, columns=ohe.get_feature_names_out(['Category'])).astype(int)
merged_df = pd.concat([df, df_new], axis='columns')
# print(merged_df)
final_df = merged_df.drop(['Category', 'Category_ham'], axis='columns')
print(final_df)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(final_df.Message, final_df.Category_spam, test_size=0.3)
from sklearn.pipeline import Pipeline

clf = Pipeline([('vectorizer', CountVectorizer()),
                ('nb', MultinomialNB())
                ])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
clf.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)
