import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from joblib import dump, load

boston = load_boston()
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df.columns = [x.lower() for x in df.columns]
df["medv"] = boston.target

X = df[[x for x in df.columns if x != 'medv']]
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scalers = [StandardScaler(), MinMaxScaler()]

pipe = Pipeline([
        ('scaler', None),
        ('regressor', RandomForestRegressor())
        ])

search_params = [
                {'scaler' : scalers,
                 'regressor' : [RandomForestRegressor()],
                 'regressor__n_estimators': [50, 100, 200],
                 'regressor__max_depth': [3, 5, 7, 9],
                 'regressor__random_state': [42] },

                {'scaler' : scalers,
                 'regressor' : [KNeighborsRegressor()],
                 'regressor__n_neighbors': [5, 10]
                 }
                 ]

search = GridSearchCV(pipe, search_params, n_jobs=-1, cv=5, verbose=2)
search.fit(X_train, y_train)

print(f"Best parameter (CV score=):{search.best_score_}")
print(search.best_params_)

print('Best Pipeline')
print(search.best_estimator_)

test_score = search.score(X_test, y_test)
print(test_score)

# Save the pipeline
dump(search.best_estimator_, 'boston_reg.joblib', compress=1)

