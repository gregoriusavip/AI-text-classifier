import tensorflow as tf
import os
import numpy as np

class text_model:
    def __init__(self):
        cwd = os.getcwd()
        relative_path = os.path.join("text-classifier-tensorflow/text_classification_model")
        absolute_path = os.path.abspath(os.path.join(cwd, relative_path))
        #If the file is not found, make sure that the absolute path is the path of text_classification_model folder
        self.loaded_model = tf.keras.models.load_model(absolute_path)

    def predict_text(self, input_text):
        predictions = self.loaded_model.predict(np.array([input_text]))
        if (predictions > 0):
            return "positive"
        elif (predictions < 0):
            return "negative"
        else:
            return "neutral"