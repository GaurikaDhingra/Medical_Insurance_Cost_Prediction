import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("insurance.csv")
print(df.head())

df.info()
df.describe()

df.nunique()
df['children'].unique() #Number of children covered by health insurance

df.isnull().sum()

##Data Handling 
labelencoder = preprocessing.LabelEncoder()

df['sex'] = labelencoder.fit_transform(df['sex'])
print(df['sex'].head())
df['sex'].unique() # 0 - female , 1-male 

df['smoker'] = labelencoder.fit_transform(df['smoker'])
df['smoker'].unique() # 1 - yes, 0 - no 

df['region'] = labelencoder.fit_transform(df['region'])
df['region'].unique() #'southwest'-3, 'southeast'-2,'northwest'-1, 'northeast'-0

##Split data 
X = df.drop(['charges', 'region'], axis =1)
y = df['charges']

X__train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##Model Training 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X__train, y_train)

y_pred = model.predict(X_test)
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('R2 Square: ', r2_score(y_test, y_pred))

##Hyperparameter tuning 
# Define parameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the RandomizedSearchCV
random_search.fit(X__train, y_train)

# Display the best parameters
print("Best Parameters:", random_search.best_params_)

# Update the model with the best estimator
model = random_search.best_estimator_

##Evaluating tuned model
# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Tuned Model - Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Tuned Model - R2 Score:", r2_score(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("Mean Cross-Validation MAE:", -cv_scores.mean())

##Save Model 
joblib.dump(model, 'insurance_cost_model.pkl')
print("Model saved as 'insurance_cost_model.pkl'")




