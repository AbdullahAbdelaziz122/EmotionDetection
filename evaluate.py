import tensorflow as tf
import data_loader

def evaluate_model(data_dir, img_size):
    images, labels = data_loader.load_images(data_dir, img_size)
    _, X_val, X_test, _, y_val, y_test = data_loader.split_data(images, labels)

    model = tf.keras.models.load_model('emotion_detection_model.h5')
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc}")

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")
