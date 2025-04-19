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
