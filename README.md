New York City Taxi Fare Prediction Model

Project Overview

This project develops a robust ensemble machine learning model to predict taxi fares in New York City using data from a Kaggle competition. The model leverages XGBoost, Random Forest, and Lasso regressors, with extensive feature engineering to incorporate temporal, spatial, and landmark-based features. By sampling 1% of a massive 55-million-row dataset, the project demonstrates scalable data processing, outlier handling, and hyperparameter tuning, aligning with the requirements of an Artificial Intelligence Development Specialist role.

Key Features





Data Preprocessing: Samples 1% of the dataset, handles outliers, and applies one-hot encoding for categorical features.



Feature Engineering: Extracts date-time components, calculates Haversine distances, and adds distances to NYC landmarks.



Model: Ensembles XGBoost, Random Forest, and Lasso regressors with weighted predictions.



Evaluation: Uses RMSE (Root Mean Squared Error) as the evaluation metric, per Kaggle competition rules.



Output: Generates submission files (submission_ensemble.csv, submission_gridsearch.csv) with predictions on test data.

Technologies Used





Python: Core programming language.



Pandas & NumPy: For data manipulation and numerical operations.



Scikit-learn: For Random Forest, Lasso, and one-hot encoding.



XGBoost: For gradient boosting model.



Opendatasets: For downloading Kaggle competition data.



Matplotlib: For visualizing model performance (e.g., overfitting curves).

Dataset

The dataset is sourced from the Kaggle competition "New York City Taxi Fare Prediction" (https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data). It includes:





Features: Pickup and dropoff coordinates, passenger count, pickup datetime, and derived features (distance, landmark distances, time-based features).



Target: Fare amount (continuous variable for regression).



Files: train.csv (55M rows), test.csv, sample_submission.csv.



Sampling: 1% (~550,000 rows) of train.csv is used for training and validation to manage computational constraints.

Installation





Clone the repository:

git clone https://github.com/your-username/NYC-Taxi-Fare-Prediction.git
cd NYC-Taxi-Fare-Prediction



Install dependencies:

pip install -r requirements.txt

The requirements.txt should include:

pandas
numpy
scikit-learn
xgboost
opendatasets
matplotlib



Install opendatasets (if not included in requirements):

pip install opendatasets --quiet

Dataset Download





Download the Kaggle dataset:





Run the following Python code to download the dataset from the Kaggle competition:

import opendatasets as od
dataset_url = 'https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data'
od.download(dataset_url)



When prompted, provide your Kaggle username and API key. Learn more about Kaggle credentials: http://bit.ly/kaggle-creds.



The dataset will be extracted to the ./new-york-city-taxi-fare-prediction directory, containing train.csv, test.csv, sample_submission.csv, and GCP-Coupons-Instructions.rtf.



Verify dataset:





Check the dataset files:

ls -lh new-york-city-taxi-fare-prediction

Expected output:

total 5.4G
-rw-r--r-- 1 root root  486 May 13 08:32 GCP-Coupons-Instructions.rtf
-rw-r--r-- 1 root root 336K May 13 08:32 sample_submission.csv
-rw-r--r-- 1 root root 960K May 13 08:32 test.csv
-rw-r--r-- 1 root root 5.4G May 13 08:33 train.csv



Verify the number of rows in train.csv:

wc -l new-york-city-taxi-fare-prediction/train.csv

Expected output: 55423856 new-york-city-taxi-fare-prediction/train.csv.



Set data directory:





Update the script to point to the correct data directory:

data_dir = 'new-york-city-taxi-fare-prediction'

Usage





Run the script:

python taxi_fare_prediction.py

Note: Ensure taxi_fare_prediction.py contains the provided code (data loading, feature engineering, model training, and ensembling).



Outputs:





Console Output: Training and validation RMSE scores for XGBoost, Random Forest, Lasso, and the ensemble model.





Example: Ensemble Validation RMSE: 3.88



Submission Files:





submission_ensemble.csv: Predictions from the weighted ensemble.



submission_gridsearch.csv: Predictions from the hyperparameter-tuned XGBoost model.



Parquet Files: train.parquet, val.parquet for processed training and validation data.



Visualizations: Overfitting curves for hyperparameter tuning (e.g., min_child_weight).

Project Structure

NYC-Taxi-Fare-Prediction/
├── taxi_fare_prediction.py      # Main script
├── new-york-city-taxi-fare-prediction/  # Dataset directory
│   ├── train.csv                # Training data (55M rows)
│   ├── test.csv                 # Test data
│   ├── sample_submission.csv    # Sample submission
│   ├── GCP-Coupons-Instructions.rtf  # Additional file
├── train.parquet                # Processed training data
├── val.parquet                  # Processed validation data
├── submission_ensemble.csv      # Output: Ensemble submission
├── submission_gridsearch.csv    # Output: GridSearch submission
├── requirements.txt             # Dependencies
└── README.md                    # This file

Feature Engineering





Date-Time Features: Extracted year, month, day, hour, day of week, day of year, and weekend status from pickup_datetime.



Time of Day: Categorized hours into Morning, Afternoon, Evening, and Night.



Haversine Distance: Calculated trip distance in kilometers using pickup and dropoff coordinates.



Landmark Distances: Added distances from dropoff points to NYC landmarks (e.g., JFK Airport, Times Square).



One-Hot Encoding: Applied to time_of_day and pickup_datetime_dayofweek for categorical features.

Model Details





XGBoost Regressor:





Best Parameters (via GridSearch): n_estimators=300, learning_rate=0.1, max_depth=7.



Performance: Train RMSE: 2.82, Val RMSE: 3.91.



Random Forest Regressor:





Parameters: n_estimators=100, random_state=42.



Performance: Train RMSE: 1.43, Val RMSE: 3.96.



Lasso Regressor:





Parameters: alpha=0.1, random_state=42.



Performance: Train RMSE: 5.19, Val RMSE: 5.28.



Ensemble:





Weighted average (60% XGBoost, 30% Random Forest, 10% Lasso, normalized).



Performance: Ensemble Validation RMSE: 3.88.

Results





The ensemble model achieves a validation RMSE of 3.88, indicating strong predictive performance on unseen data.



The submission files (submission_ensemble.csv, submission_gridsearch.csv) are formatted for Kaggle competition submission, predicting taxi fares for the test set.



Hyperparameter tuning via GridSearch improved XGBoost performance, balancing bias and variance.

Future Improvements





Incorporate additional features (e.g., traffic data, weather conditions).



Experiment with advanced ensembling techniques (e.g., stacking).



Add visualizations for feature importance and prediction errors.



Optimize data sampling to include more data while managing computational resources.

Author

MD Mehedi Hasan Mithun
9mithun9@gmail.com
https://www.linkedin.com/in/md-mehedi-hasan-mithun-1428b1124/
