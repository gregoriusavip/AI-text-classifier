import joblib
import os

class rfc_model():
    def __init__(self):
        cwd = os.getcwd()
        rfc_filepath = os.path.join(cwd, "text-classifier-tensorflow", "rfc_model", "rfc_sentiment_model.joblib")
        rfc_vectorizer = os.path.join(cwd, "text-classifier-tensorflow", "rfc_model", "rfc_vectorizer.joblib")
        rfc_encoder = os.path.join(cwd, "text-classifier-tensorflow", "rfc_model", "rfc_encoder.joblib")

        self.rfc = joblib.load(rfc_filepath)
        self.vectorizer = joblib.load(rfc_vectorizer)
        self.encoder = joblib.load(rfc_encoder)
    
    def predict(self, text):
        X_new_counts = self.vectorizer.transform([text])
        predicted = self.rfc.predict(X_new_counts)
        return self.encoder.inverse_transform(predicted)[0]