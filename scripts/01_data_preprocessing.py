import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.drop_duplicates(subset='id')
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    cat_cols = ['gender', 'part_time_job', 'extracurricular_activities', 'career_aspiration']
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    score_cols = ['math_score', 'history_score', 'physics_score',
                  'chemistry_score', 'biology_score', 'english_score', 'geography_score']
    df['total_score'] = df[score_cols].sum(axis=1)
    df['average_score'] = df['total_score'] / len(score_cols)
    le = LabelEncoder()
    for c in ['gender', 'part_time_job', 'extracurricular_activities']:
        df[c] = le.fit_transform(df[c])
    df = pd.get_dummies(df, columns=['career_aspiration'], prefix='career')
    df = df.drop(columns=['id', 'first_name', 'last_name', 'email'])
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    raw = load_data('data/student-scores.csv')
    processed = preprocess(raw)
    save_data(processed, 'data/processed/student_scores_processed.csv')

if __name__ == '__main__':
    main()
