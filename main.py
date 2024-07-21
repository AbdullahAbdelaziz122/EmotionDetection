import train
import evaluate
import predict

if __name__ == "__main__":
    # Parameters
    data_dir = 'dataset'
    img_size = 48
    epochs = 50
    batch_size = 64
    model_path = 'emotion_detection_model.h5'
    img_path = 'path_to_new_image.jpg'

    # Train the model
    train.train_model(data_dir, img_size, epochs, batch_size)

    # Evaluate the model
    evaluate.evaluate_model(data_dir, img_size)

    # Predict emotion for a new image
    emotion = predict.predict_emotion(model_path, img_path, img_size)
    print(f'Predicted Emotion: {emotion}')
