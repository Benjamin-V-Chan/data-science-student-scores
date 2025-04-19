# Load processed data
# Define X = all features except 'average_score'; y = 'average_score'
# Split into train/test sets (e.g. 80/20)
# Instantiate RandomForestRegressor (with fixed random_state)
# Train on training data
# Predict on test data
# Compute RMSE and R2; save metrics.json under outputs/metrics.json
# Extract feature_importances_; save as CSV under outputs/models/feature_importances.csv
# Serialize trained model to outputs/models/student_score_model.pkl
