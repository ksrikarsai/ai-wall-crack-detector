import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("Starting...")
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    "train",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

val = val_gen.flow_from_directory(
    "validation",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=5)

model.save("wall_crack_model.h5")

