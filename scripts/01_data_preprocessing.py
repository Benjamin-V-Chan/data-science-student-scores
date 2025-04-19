# Load raw data from data/student-scores.csv
# Drop duplicate rows based on 'id'
# Handle missing values:
#   - Fill numeric columns with their median
#   - Fill categorical columns with their mode
# Feature engineering:
#   - Create 'total_score' = sum of all subject scores
#   - Create 'average_score' = total_score / number of subjects
# Encode categorical variables:
#   - Label‑encode 'gender', 'part_time_job', 'extracurricular_activities'
#   - One‑hot encode 'career_aspiration'
# Drop personal‑info columns: 'id','first_name','last_name','email'
# Save processed DataFrame to data/processed/student_scores_processed.csv
