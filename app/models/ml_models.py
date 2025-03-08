import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_and_save_model():
    # Sample dataset
    x = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    # Training forest classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x, y)

    # Save model
    joblib.dump(model, 'model.joblib')
    return model


def get_model():
    try:
        model = joblib.load('model.joblib')
    except (FileNotFoundError, OSError):
        model = train_and_save_model()
    return model


# Function to make predictions
def predict(features):
    model = get_model()
    prediction = model.predict([features])
    return prediction[0]
