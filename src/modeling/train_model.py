import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit

output_dir = 'models'

# Create directory structure if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loads X and y data
data = pd.read_csv('Data/processed/processed_game_data_2021-26.csv')
y = data['Target Data'].copy()
X = data.drop(columns=['Target Data'], axis=1).copy()

# Creates training and test data
# Because the data is a time series the data is split into the first 80 and last 20 percent to avoid target leakage
split_point = int(len(X)*0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

# Creation of model
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Preforms 5-fold cross validation using TimeSeriesSplit to prevent target leakage
tscv = TimeSeriesSplit(n_splits=5)
crossVal = cross_val_score(model, X, y, cv=tscv)

# Fits the model and creates predictions and accuracy score for the test data
model.fit(X_train, y_train)
predictions_test = model.predict(X_test)
score_test = accuracy_score(y_test, predictions_test)

# Creates accuracy score for training data
predictions_train = model.predict(X_train)
score_train = accuracy_score(y_train, predictions_train)

# Displays model statistics
print(f'CV Score {crossVal.mean():.2f}')
print(f'Test Score {score_test:.2f}')
print(f'Training Score {score_train:.2f}')
print(f'baseline {y[y==1].sum()/len(y):.2f}')

# Saves the model
joblib.dump(model, os.path.join(output_dir, 'final_model.joblib'))
