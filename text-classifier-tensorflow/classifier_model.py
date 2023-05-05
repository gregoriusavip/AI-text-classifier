import tensorflow as tf
import os
import numpy as np

class text_model:
    cwd = os.getcwd()
    relative_path = os.path.join("text_classification_model")
    absolute_path = os.path.abspath(os.path.join(cwd, relative_path))
    #If the file is not found, make sure that the absolute path is the path of text_classification_model folder
    loaded_model = tf.keras.models.load_model(absolute_path)

    def predict_text(self, input_text):
        predictions = self.loaded_model.predict(np.array([input_text]))
        return predictions

model = text_model()