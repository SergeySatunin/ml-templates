import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def get_xgb_baseline(features_dataframe, target_dataframe, cv_fold: int, search_params: dict) -> dict:
    """
    :param search_params: dictionary of parameters for grid search, e.g.
    parameters = {'learning_rate': [0.01],
                  'gamma': [1],
                  'max_depth': [4, 5],
                  'min_child_weight': [0],
                  'subsample': [0.7, 0.8, 1],
                  'colsample_bytree': [0.8],
                  'n_estimators': [50, 100, 150],
                  'scale_pos_weight': [1],
                  'seed': [123]}
    """

    xgb_model = xgb.XGBClassifier()
    search_params['objective'] = ['binary:logistic']

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=search_params,
                               scoring='roc_auc', cv=cv_fold, verbose=2)

    grid_search.fit(features_dataframe, target_dataframe)

    optimized_parameters = grid_search.best_params_
    cv_auc = grid_search.best_score_
    model = grid_search.best_estimator_
    importance_scores = model.get_booster().get_score(importance_type='gain')

    predicted_binary = model.predict(features_dataframe)

    conf_matrix = confusion_matrix(target_dataframe, predicted_binary)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return {'parameters': optimized_parameters,
            'cv_auc': cv_auc,
            'model': grid_search.best_estimator_,
            'feature_importance': importance_scores,
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1}
