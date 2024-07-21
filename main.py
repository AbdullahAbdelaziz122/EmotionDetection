import train
import evaluate
import predict
from data_loader import load_images, split_data
from train import train_model
from predict import predict_emotion
from evaluate import evaluate_model




if __name__ == "__main__":
    # Paths
    train_dir = '/content/EmotionDetection/archive/train'
    test_dir = '/content/EmotionDetection/archive/test'
    img_size = 48

    # Load training and test data
    X_train, y_train = load_images(train_dir, img_size)
    X_test, y_test = load_images(test_dir, img_size)

    # Parameters
    data_dir = '/content/EmotionDetection/archive'
    img_size = 48
    epochs = 50
    batch_size = 64

    # Train the model

    history = train_model(  
                            data_dir,
                            img_size,
                            epochs,
                            batch_size,
                            X_train,
                            X_test,
                            y_train,
                            y_test
                            )

    #Evaluate the model
    val_acc, test_acc = evaluate_model()


    

    # Predict on a new image
    model_path = 'emotion_detection_model_h5'
    img_path = 'test.jpg'
    img_size = 48
    emotion = predict_emotion(model_path, img_path, img_size)
    print(f"Predicted emotion: {emotion}")