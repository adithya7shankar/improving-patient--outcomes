import pandas as pd
import tensorflow as tf
# File paths
file_paths = [
    "Datasets/Disease_symptom_and_patient_profile_dataset.csv",
    "Datasets/healthcare_dataset.csv",
    "Datasets/medquad.csv",
    "Datasets/readmissions-for-isolated-coronary-artery-bypass-graft-cabg-complications-metadata.csv"
]

# Load datasets
datasets = {file_path.split('/')[-1]: pd.read_csv(file_path) for file_path in file_paths}

print("Datasets loaded successfully!")
# Display the datasets to the user
datasets.keys()
df_disease_symptom = pd.read_csv("Datasets/Disease_symptom_and_patient_profile_dataset.csv")
df_healthcare_dataset = pd.read_csv("Datasets/healthcare_dataset.csv")
df_medquad = pd.read_csv("Datasets/medquad.csv")

df_coronary_complications = pd.read_csv("Datasets/readmissions-for-isolated-coronary-artery-bypass-graft-cabg-complications-metadata.csv")

# Load the dataset
datset_list = [
df_disease_symptom ,
df_healthcare_dataset,
df_medquad,
df_coronary_complications
]

for dataset in datset_list:
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())

# The dataset contains information about diseases, symptoms, patient profiles, and an outcome variable. The columns include:

### Disease: The type of disease (e.g., Influenza, Common Cold).
# Fever, Cough, Fatigue, Difficulty Breathing: Symptoms marked as "Yes" or "No."
# Age: The age of the patient.
#Gender: The gender of the patient.
#Blood Pressure: Blood pressure status (e.g., Low, Normal).
#Cholesterol Level: Cholesterol level (e.g., Normal).
#Outcome Variable: The target variable indicating the outcome, marked as "Positive" or "Negative."
#To create a neural network, we'll need to:

#Convert categorical variables (like symptoms, gender, etc.) into numerical values.
#Normalize the numerical variables (like age).
#Split the data into training and test sets.#
#Build and train a neural network model.


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Convert categorical variables to numerical values
label_encoders = {}
categorical_columns = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
                       'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']

for column in categorical_columns:
    le = LabelEncoder()
    df_disease_symptom[column] = le.fit_transform(df_disease_symptom[column])
    label_encoders[column] = le

# Normalize the numerical values
scaler = StandardScaler()
df_disease_symptom['Age'] = scaler.fit_transform(df_disease_symptom[['Age']])

# Split the data into input features (X) and target variable (y)
X = df_disease_symptom.drop('Outcome Variable', axis=1)
y = df_disease_symptom['Outcome Variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

test_loss, test_accuracy












##NN 2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Drop irrelevant columns
columns_to_drop = ['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date']
dataset_cleaned = df_healthcare_dataset.drop(columns=columns_to_drop)

# Convert categorical variables to numerical values
label_encoders = {}
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']

for column in categorical_columns:
    le = LabelEncoder()
    dataset_cleaned[column] = le.fit_transform(dataset_cleaned[column])
    label_encoders[column] = le

# Check for any missing values
dataset_cleaned = dataset_cleaned.dropna()

# Split the data into input features (X) and target variable (y)
X = dataset_cleaned.drop('Billing Amount', axis=1)
y = dataset_cleaned['Billing Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae, rmse
