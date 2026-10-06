import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]


def predict(img_path):

    # Load image
    image = Image.open(img_path)

    # Resize image to model input size
    image = image.resize((180, 180))

    # Convert image to NumPy array
    image = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Normalize
    image = image / 255.0

    # Prediction
    prediction = model.predict(image, verbose=0)

    # Get predicted class index
    predicted_index = np.argmax(prediction[0])

    # Get predicted class
    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = prediction[0][predicted_index]

    return predicted_class, confidence