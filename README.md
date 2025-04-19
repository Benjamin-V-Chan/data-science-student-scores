# data-science-student-scores

# Project Overview

This project analyzes the end-of-semester performance of senior students at a large high school. It includes steps to preprocess raw scores data, explore key patterns, train a model to predict average scores, simulate the impact of varying study and absence levels, and generate a final report summarizing findings.

# Folder Structure

```
project-root/
├── data/
│   ├── student-scores.csv          # Raw dataset
│   └── processed/
│       └── student_scores_processed.csv  # Cleaned and feature-engineered data
├── scripts/
│   ├── 01_data_preprocessing.py    # Data cleaning & feature engineering
│   ├── 02_exploratory_analysis.py  # Descriptive stats & visualizations
│   ├── 03_model_training.py        # Model fitting & evaluation
│   ├── 04_simulation_analysis.py   # Scenario-based predictions
│   └── 05_report_generation.py     # Report assembly
└── outputs/
    ├── descriptive_stats.csv        # Summary statistics
    ├── figures/                     # All generated plots
    ├── metrics.json                 # Model performance metrics
    ├── models/                      # Saved model & importances
    ├── simulation_results.csv       # Simulation outcomes
    └── report.json                  # Final summary report
```

# Usage

1. **Setup the Project:**
   - Clone the repository.
   - Ensure you have Python installed.
   - Install required dependencies using the requirements.txt file.
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data:**
   ```bash
   python scripts/01_data_preprocessing.py
   ```

3. **Run Exploratory Analysis:**
   ```bash
   python scripts/02_exploratory_analysis.py
   ```

4. **Train Prediction Model:**
   ```bash
   python scripts/03_model_training.py
   ```

5. **Perform Simulation Analysis:**
   ```bash
   python scripts/04_simulation_analysis.py
   ```

6. **Generate Final Report:**
   ```bash
   python scripts/05_report_generation.py
   ```

# Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

# Acknowledgments

dataset name: 🎓 Student Scores  
dataset author: mexwell  
dataset source: https://www.kaggle.com/datasets/mexwell/student-scores

