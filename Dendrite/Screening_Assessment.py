import json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

def parse_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def load_data(csv_file):
    return pd.read_csv(csv_file)

def impute_missing_values(df, feature_params):
    for feature in feature_params:
        column = feature['name']
        imputation_method = feature['imputation']
        if imputation_method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif imputation_method == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            continue
        df[column] = imputer.fit_transform(df[[column]])
    return df

def feature_reduction(df, target, reduction_method):
    X = df.drop(columns=[target])
    y = df[target]
    if reduction_method == 'No Reduction':
        return X, y
    elif reduction_method == 'Corr with Target':
        selector = SelectKBest(score_func=f_regression, k='all')
        X_new = selector.fit_transform(X, y)
        return pd.DataFrame(X_new, columns=X.columns[selector.get_support()]), y
    elif reduction_method == 'PCA':
        pca = PCA(n_components=0.95)
        X_new = pca.fit_transform(X)
        return pd.DataFrame(X_new), y
    else:
        return X, y

def create_model(prediction_type, model_params):
    if prediction_type == 'regression':
        model_options = {
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor()
        }
    else:
        raise ValueError('Unsupported prediction type')
    
    models = []
    for model_param in model_params:
        if model_param['is_selected']:
            model = model_options[model_param['name']]
            param_grid = model_param['hyper_params']
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
            models.append(grid_search)
    return models

def main():
    json_data = parse_json('algoparams_from_ui.json')
    
    target = json_data['target']
    prediction_type = json_data['prediction_type']
    feature_params = json_data['features']
    reduction_method = json_data['feature_reduction']
    model_params = json_data['models']
    
    df = load_data('data.csv')
    df = impute_missing_values(df, feature_params)
    X, y = feature_reduction(df, target, reduction_method)
    
    models = create_model(prediction_type, model_params)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        print(f"Model: {model.estimator.__class__.__name__}")
        print(f"Best Parameters: {model.best_params_}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 Score: {r2_score(y_test, y_pred)}")

if __name__ == '__main__':
    main()
