import os
import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def load_data(path):
    return pd.read_csv(path)

def run_simulation(df, model, feature, increments):
    results = []
    base = df.copy()
    for inc in increments:
        temp = base.copy()
        temp[feature] = temp[feature] + inc
        preds = model.predict(temp.drop(columns=['average_score']))
        results.append({'feature': feature,
                        'increment': inc,
                        'mean_predicted_score': preds.mean()})
    return results

def main():
    df = load_data('data/processed/student_scores_processed.csv')
    model = load_model('outputs/models/student_score_model.pkl')
    sims = []
    sims += run_simulation(df, model, 'weekly_self_study_hours', [0, 5, 10])
    sims += run_simulation(df, model, 'absence_days', [0, 5, 10])
    sim_df = pd.DataFrame(sims)
    os.makedirs('outputs', exist_ok=True)
    sim_df.to_csv('outputs/simulation_results.csv', index=False)

if __name__ == '__main__':
    main()
