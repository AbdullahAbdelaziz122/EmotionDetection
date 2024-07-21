train_dir = './EmotionDetection/archive/train'
test_dir = './EmotionDetection/archive/test'
img_size = 48
## Load images from the data directory

def load_images(data_dir, img_size):
    images = []
    labels = []
    label_map = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'neutral': 6}

    for label in label_map:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            print(f"Directory {label_dir} does not exist.")
            continue
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label_map[label])
            else:
                print(f"Failed to load image: {img_path}")

    images = np.array(images) / 255.0  # Normalize pixel values
    images = np.expand_dims(images, -1)  # Add a channel dimension
    labels = np.array(labels)

    return images, labels

def split_data(X_train, y_train):
    X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_val, X_test, y_val, y_test


# Load training and test data
X_train, y_train = load_images(train_dir, img_size)
X_test, y_test = load_images(test_dir, img_size)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
