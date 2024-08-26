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

# Display the datasets to the user
datasets.keys()
