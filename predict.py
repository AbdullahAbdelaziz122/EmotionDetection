import tensorflow as tf
import cv2
import numpy as np

def predict_emotion(model_path, img_path, img_size):
    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    prediction = model.predict(img)
    emotion = np.argmax(prediction)

    emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return emotion_dict[emotion]
