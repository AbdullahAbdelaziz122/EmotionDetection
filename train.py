def train_model(data_dir, img_size, epochs, batch_size, X_train, X_test, y_train, y_test):
  """
  Trains a CNN model for emotion detection.

  Args:
      data_dir: Path to the directory containing the image data (unused in this case).
      img_size: Desired image size for preprocessing.
      epochs: Number of training epochs.
      batch_size: Batch size for training.
      X_train: Training images.
      X_test: Testing images.
      y_train: Training labels.
      y_test: Testing labels.

  Returns:
      The training history object.
  """

  input_shape = (img_size, img_size, 1)
  cnn_model = create_model(input_shape)  # Assuming create_model is defined elsewhere

  history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

  cnn_model.save(f'emotion_detection_model_h5')  # Consider including date in filename
  return history
