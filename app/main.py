import pandas as pd

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