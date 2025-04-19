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
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('outputs/figures/correlation_heatmap.png')
    plt.close()
    for cat in ['gender', 'part_time_job', 'extracurricular_activities']:
        plt.figure()
        sns.boxplot(x=cat, y='average_score', data=df)
        plt.title(f'Average Score by {cat}')
        plt.savefig(f'outputs/figures/box_{cat}.png')
        plt.close()

if __name__ == '__main__':
    main()
