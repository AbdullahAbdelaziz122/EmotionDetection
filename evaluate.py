import tensorflow as tf

def evaluate_model():
    X_train, y_train = load_images(train_dir, img_size)
    X_test, y_test = load_images(test_dir, img_size)
    X_val, X_test, y_val, y_test = split_data(X_train, y_train)
    input_shape = (img_size, img_size, 1)
    cnn_model = create_model(input_shape)  # Assuming create_model is defined elsewhere
    cnn_model.load_weights('emotion_detection_model_h5')
    val_loss, val_acc = cnn_model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc}")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc}")
    return val_acc, test_acc