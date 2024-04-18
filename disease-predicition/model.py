import pandas as pd
import numpy as np
import pickle

with open('./Disease-Pred-Model.sav', 'rb') as f:
    model = pickle.load(f)

# List of all features
features = ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision',
            'excessive_hunger', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
            'movement_stiffness', 'depression', 'irritability', 'visual_disturbances',
            'painful_walking', 'abdominal_pain', 'nausea', 'vomiting', 'blood_in_mucus',
            'Fatigue', 'Fever', 'Dehydration', 'loss_of_appetite', 'cramping',
            'blood_in_stool', 'gnawing', 'upper_abdomain_pain', 'fullness_feeling',
            'hiccups', 'abdominal_bloating', 'heartburn', 'belching', 'burning_ache']

# Input data dictionary
input_data = {
    "acidity": 1,
    "headache": 0,
    "excessive_hunger": 0,
    "muscle_weakness": 1,
    "stiff_neck": 0,
    "swelling_joints": 1,
    "movement_stiffness": 0,
    "depression": 1,
    "irritability": 0,
    "visual_disturbances": 1,
    "painful_walking": 0,
    "abdominal_pain": 1,
    "nausea": 0,
    "vomiting": 1,
    "blood_in_mucus": 0,
    "Fatigue": 1,
    "Fever": 0,
    "Dehydration": 1,
    "loss_of_appetite": 0,
    "cramping": 1,
    "blood_in_stool": 0,
    "gnawing": 1,
    "upper_abdomain_pain": 0,
    "fullness_feeling": 1,
    "hiccups": 0,
    "abdominal_bloating": 1,
    "heartburn": 0,
    "belching": 1,
    "burning_ache": 0,
    "indigestion": 1,
    "blurred_and_distorted_vision": 1
}

# Find the missing feature
missing_feature = [feature for feature in features if feature not in input_data]
print("Missing feature:", missing_feature)

def preprocess_input_data(input_data):
    # Convert the input data into a DataFrame
    df = pd.DataFrame(input_data, index=[0])
    return df


input_df = preprocess_input_data(input_data)

predictions = model.predict(input_df)
print("Predicted diseases:", predictions)