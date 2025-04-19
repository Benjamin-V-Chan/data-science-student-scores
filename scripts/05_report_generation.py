import json
import pandas as pd
import os

def load_metrics(path):
    return json.load(open(path))

def load_importances(path):
    imp = pd.read_csv(path, index_col=0)
    imp.columns = ['importance']
    return imp

def load_simulation(path):
    return pd.read_csv(path)

def main():
    os.makedirs('outputs', exist_ok=True)
    metrics = load_metrics('outputs/metrics.json')
    imp = load_importances('outputs/models/feature_importances.csv')
    top5 = imp.sort_values('importance', ascending=False).head(5)
    sims = load_simulation('outputs/simulation_results.csv')
    report = {
        'model_performance': metrics,
        'top_features': top5['importance'].to_dict(),
        'simulation_summary': sims.head(10).to_dict(orient='records')
    }
    with open('outputs/report.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
