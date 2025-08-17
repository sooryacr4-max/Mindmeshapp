import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv('mesh_prompt_dataset_extended_250.csv')

# Separate features and targets
X = df['Prompt']
Y = df[['Geom']]

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=43, test_size=0.10, shuffle=True)

# Vectorizer + model in a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultiOutputClassifier(RandomForestClassifier()))
])

# Train
pipeline.fit(X_train, Y_train)

# Save model for later use
joblib.dump(pipeline, 'mindmesh_model.pkl')


