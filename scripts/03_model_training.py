import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def train_and_evaluate(df):
    X = df.drop(columns=['average_score'])
    y = df['average_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return model, {'rmse': rmse, 'r2': r2}

def save_artifacts(model, metrics):
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(model, 'outputs/models/student_score_model.pkl')
    pd.Series(model.feature_importances_,
              index=model.feature_names_in_).sort_values(ascending=False) \
        .to_csv('outputs/models/feature_importances.csv', header=['importance'])
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    df = load_data('data/processed/student_scores_processed.csv')
    model, metrics = train_and_evaluate(df)
    save_artifacts(model, metrics)

if __name__ == '__main__':
    main()
