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

# Load the dataset
df_disease_symptom = pd.read_csv("Datasets/Disease_symptom_and_patient_profile_dataset.csv")

# Display the first few rows
print(df_disease_symptom.head())

# Get basic information
print(df_disease_symptom.info())