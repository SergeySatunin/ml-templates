import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from matplotlib import pyplot as plt
import shap

def get_rf_baseline(features_dataframe, target_column, cv_fold: int, search_params: dict, score) -> dict:

    rf = RandomForestRegressor()

    X_train = features_dataframe
    Y_train = target_column

    grid_search = GridSearchCV(estimator=rf, param_grid=search_params, scoring=score, cv=cv_fold, verbose=2)

    grid_search.fit(X_train, Y_train)

    importance_scores = grid_search.best_estimator_.feature_importances_
    feature_importance = sorted(zip(features_dataframe.columns, importance_scores), key=lambda x: x[1], reverse=True)

    feature_names = [x[0] for x in feature_importance]
    sorted_scores = [x[1] for x in feature_importance]

    return {'parameters': grid_search.best_params_,
            'cv_auc': grid_search.best_score_,
            'model': grid_search.best_estimator_,
            'feature_names' : feature_names,
            'importance_scores': sorted_scores,
            }

def main():
    boston = load_boston()
    df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
    df.columns = [x.lower() for x in df.columns]
    df["medv"] = boston.target

    features = [x for x in df.columns if x != 'medv']
    target = 'medv'
    score = 'neg_root_mean_squared_error'


    search_params = {'n_estimators': [50, 100, 200],
                     'max_depth': [3, 5, 7, 9],
                     'random_state': [42] }


    baseline = get_rf_baseline(df[features], df[target], 5, search_params, score)
    plt.barh(baseline['feature_names'], baseline['importance_scores'])

    explainer = shap.TreeExplainer(baseline['model'])
    shap_values = explainer.shap_values(df[features])
    shap.summary_plot(shap_values, df[features], plot_type="bar")

if __name__ == '__main__':
    main()