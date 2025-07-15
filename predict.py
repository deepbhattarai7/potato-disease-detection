import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Directory for storing uploaded images
uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Function to load training history
def load_training_history(history_path='training_history.npy'):
    try:
        with open(history_path, 'rb') as f:
            history = np.load(f, allow_pickle=True).item()
        return history
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

# Function to load and preprocess an image
def load_image(image_path, target_size=(128, 128)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error: Could not read image '{image_path}'. Check file path/integrity.")
        img = cv2.resize(img, target_size)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0 
        return img, None
    except Exception as e:
        return None, f"Error loading image: {str(e)}"

# Function to predict disease from image
def predict_image(image_path, model_path='best_model.keras'):
    try:
        model = load_model(model_path)
        img, load_error = load_image(image_path)
        if img is None:
            return None, None, load_error
        predictions = model.predict(img)
        class_names = ['Potato__Early_blight', 'Potato_healthy', 'Potato__Late_blight']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = predictions[0]
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, f"Error during prediction: {str(e)}"

# Streamlit main function
def main():
    st.title('Potato Leaf Disease Detection')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(file_path, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction, confidence, error = predict_image(file_path)
                if error:
                    st.error(f"Prediction Error: {error}")
                else:
                    st.success(f"Predicted class: {prediction}")
                    st.write("Confidence scores:")
                    confidence_df = pd.DataFrame(confidence, index=['Potato__Early_blight', 'Potato_healthy',
                                                                    'Potato__Late_blight'], columns=['Confidence'])
                    st.dataframe(confidence_df)

    # Show training history
    if st.checkbox('Show Training History'):
        history = load_training_history()
        if history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history['accuracy'], label='Train Accuracy')
            ax1.plot(history['val_accuracy'], label='Val Accuracy')
            ax1.set_title('Accuracy over Epochs')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend()

            ax2.plot(history['loss'], label='Train Loss')
            ax2.plot(history['val_loss'], label='Val Loss')
            ax2.set_title('Loss over Epochs')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend()

            st.pyplot(fig)

    # Show confusion matrix and classification report
    if st.checkbox('Show Confusion Matrix & Classification Report'):
        try:
            X_test = np.load('X_test.npy')
            y_test = np.load('y_test.npy')
            y_test_categorical = pd.get_dummies(y_test).values
            model = load_model('best_model.keras')
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test_categorical, axis=1)
            cm = confusion_matrix(y_true, y_pred_classes)
            acc = accuracy_score(y_true, y_pred_classes)

            st.write(f"### Model Accuracy: {acc:.2f}")

            # Generate classification report as a dataframe
            report_dict = classification_report(y_true, y_pred_classes, target_names=['Early Blight', 'Healthy', 'Late Blight'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            # Display classification report in a table
            st.write("### Classification Report")
            st.dataframe(report_df)

            # Display confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Early Blight', 'Healthy', 'Late Blight'],
                        yticklabels=['Early Blight', 'Healthy', 'Late Blight'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error displaying Confusion Matrix & Classification Report: {e}")

    st.text("Please upload an image file (JPEG or PNG) to predict the disease on potato leaves.")

if __name__ == '__main__':
    main()
