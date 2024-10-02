import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt


def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, 3, activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(512, 3, activation="relu", padding="same")(c5)

    # Upsampling
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(256, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(128, 3, activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(64, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(32, 3, activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Assuming you have X_train and y_train ready as your dataset
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.1)


def predict_and_plot(model, image):
    image_resized = cv2.resize(image, (256, 256)) / 255.0
    image_input = np.expand_dims(image_resized, axis=0)

    prediction = model.predict(image_input)
    prediction_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction_mask, cmap="gray")
    plt.axis("off")

    plt.show()


# Example usage
image = cv2.imread("path/to/image.jpg")
predict_and_plot(model, image)
