import tensorflow as tf
import os
import numpy as np

cwd = os.getcwd()
relative_path = os.path.join("text_classification_model")

absolute_path = os.path.abspath(os.path.join(cwd, relative_path))
print(absolute_path)
loaded_model = tf.keras.models.load_model(absolute_path)

sample_text = "The movie was not good. The animation and the graphics 'were terrible. I would not recommend this movie."
predictions = loaded_model.predict(np.array([sample_text]))
print(predictions)