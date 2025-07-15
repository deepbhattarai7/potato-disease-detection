import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_data(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                resized_img = cv2.resize(img_array, (128, 128))
                data.append([resized_img, label])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return data

def preprocess_data(data):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    dataset_path = 'C:/Datasets'  
    categories = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

    data = load_data(dataset_path, categories)
    X, y = preprocess_data(data)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
