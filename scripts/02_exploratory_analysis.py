import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    return pd.read_csv(path)

def main():
    df = load_data('data/processed/student_scores_processed.csv')
    os.makedirs('outputs/figures', exist_ok=True)
    df.describe().to_csv('outputs/descriptive_stats.csv')
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'outputs/figures/hist_{col}.png')
        plt.close()
    corr = df[num_cols].corr()
