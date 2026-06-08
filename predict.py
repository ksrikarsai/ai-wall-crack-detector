import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("wall_crack_model.h5")

img = Image.open("sample2.jpeg").resize((224,224))
img = np.array(img)/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

print("Prediction Score:", pred)

if pred > 0.5:
    print("No Crack")
else:
    print("Crack Detected")