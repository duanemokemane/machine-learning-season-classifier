'''
IMPORT ALL NECESSARY LIBRARIES
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


'''
GENERATE TOY DATASET
'''


# Function to generate toy dataset for each season
def generate_season_data(season, n_samples=100):
    # Characteristic ranges for each season
    #based on celcius and based my temperature ranges from: https://study.com/academy/lesson/temperature-clouds-wind-humidity-on-the-atmospheric-cycle.html
    season_ranges = {
        "Spring": {'min_temp': (10, 20), 'max_temp': (20, 30), 'precipitation': (0, 10), 'humidity': (40, 60), 'wind_speed': (5, 15)},
        "Summer": {'min_temp': (20, 30), 'max_temp': (30, 40), 'precipitation': (0, 20), 'humidity': (50, 70), 'wind_speed': (10, 20)},
        "Autumn": {'min_temp': (10, 20), 'max_temp': (20, 30), 'precipitation': (0, 15), 'humidity': (50, 70), 'wind_speed': (5, 15)},
        "Winter": {'min_temp': (-10, 5), 'max_temp': (0, 10), 'precipitation': (0, 10), 'humidity': (40, 60), 'wind_speed': (5, 15)}
    }
    
    # Generate data for the specified seasons based on their characteristics
    #np.random.uniform generates random uniform distributions of my season ranges
    # asterisk 
    data = {
        'Min Temp Recorded': np.random.uniform(*season_ranges[season]['min_temp'], size=n_samples),
        'Max Temp Recorded': np.random.uniform(*season_ranges[season]['max_temp'], size=n_samples),
        'Precipitation': np.random.uniform(*season_ranges[season]['precipitation'], size=n_samples),
        'Humidity': np.random.uniform(*season_ranges[season]['humidity'], size=n_samples),
        'Wind Speed': np.random.uniform(*season_ranges[season]['wind_speed'], size=n_samples),
        'Season': [season] * n_samples
    }

    return pd.DataFrame(data)


'''
DATA CLEANING
'''


# Generate toy dataset for each season
spring_data = generate_season_data("Spring")
summer_data = generate_season_data("Summer")
autumn_data = generate_season_data("Autumn")
winter_data = generate_season_data("Winter")

# concatenate the datasets
df = pd.concat([spring_data, summer_data, autumn_data, winter_data], ignore_index=True)


'''
DATA CLEANING
'''


# Split the dataset into feature (X) and target (y)
X = df.drop(columns=['Season'])
y = df['Season']


'''
CREATING RFC
'''


# Create the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform cross-validation, 5 folds
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

#print the 5 CV scores
print("\nCross-validation scores:", cv_scores)
#print the mean cross validation accuracy
print("Mean CV accuracy:", np.mean(cv_scores))

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RFC
rf_classifier.fit(X_train, y_train)

# Predicting the seasons for the test set
y_pred = rf_classifier.predict(X_test)


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of my model:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



'''
DUMPING MACHINE LEARNING MODEL
'''


import joblib
# Save the trained model (as pickle(pkl)) to a file using joblib
joblib.dump(rf_classifier, r'random_forest_model.pkl')


'''
Prediction run for Question 3
'''
# Slice the test set to include only the first 5 days
X_slice = X_test[:5]
y_slice_true = y_test[:5]

# Predict the seasons for the subset
y_slice_pred = rf_classifier.predict(X_slice)

# Print the predictions and true labels for the slice
for i in range(len(y_slice_pred)):
    print(f"Random Day {i+1}: Predicted Season - {y_slice_pred[i]}, True Season - {y_slice_true.iloc[i]}")








