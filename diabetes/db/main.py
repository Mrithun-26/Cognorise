import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
df['diabetes'].value_counts()
df.isnull().sum()
df['smoking_history'].value_counts()
# convert no info into NaN value
df['smoking_history']=df['smoking_history'].replace('No Info', np.nan)


def SmokingHistoryImpute(data, column):
    # Create a boolean mask for missing values in the column
    mask = data[column].isnull()

    # Count the number of missing values
    num_missing = mask.sum()

    # If there are missing values, sample non-null values from the column
    if num_missing > 0:
        # Sample values from non-null entries in the column
        random_sample = data[column].dropna().sample(num_missing, replace=True)

        # Assign these random values to the missing values in the column
        data.loc[mask, column] = random_sample.values
SmokingHistoryImpute(df, 'smoking_history')

X = df.drop(columns=['diabetes'])
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train['smoking_history'].value_counts()
oe_order = ['never','former','not current','current','ever']
preprocessor = ColumnTransformer(transformers=[
    ('oe',OrdinalEncoder(categories=[oe_order]),['smoking_history']),
    ('ohe', OneHotEncoder(drop='first'),['gender']),
    ('scaaler',MinMaxScaler(),['age','bmi','blood_glucose_level','HbA1c_level'])
],remainder='passthrough')
X_train_trf = preprocessor.fit_transform(X_train)
X_test_trf = preprocessor.transform(X_test)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_trf, y_train)
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
clf = RandomForestClassifier(n_estimators=500,min_samples_split=6,max_samples=0.75,max_depth=30,bootstrap=True,n_jobs=-1)
clf.fit(X_train_resampled,y_train_resampled)
y_pred = clf.predict(X_test_trf)
f1_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
from sklearn.model_selection import cross_val_score
cross_val_score(clf,X_train_resampled,y_train_resampled,scoring='f1',cv=5,verbose=2,n_jobs=-1)
param_dist = {
    'n_estimators': [200,300,400,500],  # Randomly pick from 100, 150, 200, 250, 300
    'max_depth': [None, 10, 20, 30],          # Randomly pick from these values
    'min_samples_split': np.arange(2, 11, 2), # Randomly pick from 2, 4, 6, 8, 10
    'max_samples':[0.25,0.5,0.75,1],      # Randomly pick from 1, 2, 3, 4
    'bootstrap': [True, False]                # Randomly pick True or False
}
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=clf,
                                   param_distributions=param_dist,
                                   n_iter=25,              # Number of parameter settings sampled
                                   scoring='f1',            # Optimize for F1 score
                                   cv=5,                    # 5-fold cross-validation
                                   random_state=42,         # For reproducibility
                                   n_jobs=-1)               # Use all available cores
random_search.fit(X_train_resampled,y_train_resampled)
print("Best parameters found: ", random_search.best_params_)
print("Best F1 score: ", random_search.best_score_)







