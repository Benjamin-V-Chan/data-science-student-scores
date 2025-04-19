# Load processed data and the trained model
# Define two sets of scenarios:
#   1) weekly_self_study_hours + [0, 5, 10]
#   2) absence_days + [0, 5, 10]
# For each scenario group and increment:
#   - Create a copy of df with that feature increased
#   - Predict average_score
#   - Compute mean predicted average_score
# Record results in a DataFrame with columns [feature, increment, mean_predicted_score]
# Save to outputs/simulation_results.csv
