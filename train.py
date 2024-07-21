import data_loader
import model

def train_model(data_dir, img_size, epochs, batch_size):
    images, labels = data_loader.load_images(data_dir, img_size)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, labels)

    input_shape = (img_size, img_size, 1)
    cnn_model = model.create_model(input_shape)

    history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    cnn_model.save('emotion_detection_model.h5')
    return history
    