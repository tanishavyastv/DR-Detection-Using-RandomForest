import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📌 Step 1: Define Paths
DATASET_PATH = r"D:\bvp\SEM-6\DV\project\IDRiD_Dataset\B. Disease Grading\1. Original Images"
LABELS_TRAIN_PATH = r"D:\bvp\SEM-6\DV\project\IDRiD_Dataset\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
LABELS_TEST_PATH = r"D:\bvp\SEM-6\DV\project\IDRiD_Dataset\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"

# 📌 Step 2: Load Labels and Convert to Binary Classification
def load_labels(labels_path):
    labels_df = pd.read_csv(labels_path)
    labels_df['Binary_Class'] = labels_df['Retinopathy grade'].apply(lambda x: 0 if x == 0 else 1)  # Normal = 0, DR = 1
    return labels_df

labels_train = load_labels(LABELS_TRAIN_PATH)
labels_test = load_labels(LABELS_TEST_PATH)

# 📌 Step 3: Feature Extraction using GLCM
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return [contrast, dissimilarity, homogeneity, energy, correlation]

# 📌 Step 4: Preprocess Images (Resize, Green Channel, CLAHE, Extract Features)
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256))  # Resize image

    green_channel = image[:, :, 1]  # Extract Green Channel
    
    # Apply CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(green_channel)
    
    # Convert back to 3-channel image for feature extraction
    enhanced_image = cv2.merge([enhanced_image, enhanced_image, enhanced_image])

    # Extract Features
    features = extract_features(enhanced_image)

    return enhanced_image, features

# 📌 Step 5: Load Images and Extract Features
def load_dataset(images_folder, labels_df):
    X = []
    y = []
    
    for _, row in labels_df.iterrows():
        img_name = row['Image name'] + ".jpg"
        img_path = os.path.join(images_folder, img_name)

        if os.path.exists(img_path):
            _, features = preprocess_image(img_path)
            X.append(features)
            y.append(row['Binary_Class'])
        else:
            print(f"Image not found: {img_path}")

    return np.array(X), np.array(y)

# Load Training and Testing Data
X_train, y_train = load_dataset(os.path.join(DATASET_PATH, "a. Training Set"), labels_train)
X_test, y_test = load_dataset(os.path.join(DATASET_PATH, "b. Testing Set"), labels_test)

# 📌 Step 6: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 📌 Step 7: Make Predictions
y_pred = rf_model.predict(X_test)

# 📌 Step 8: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Normal", "Diabetic Retinopathy"])

# Compute Metrics
TP = conf_matrix[1, 1]  # True Positives
TN = conf_matrix[0, 0]  # True Negatives
FP = conf_matrix[0, 1]  # False Positives
FN = conf_matrix[1, 0]  # False Negatives

sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) != 0 else 0

# 📌 Step 9: Print Results
print("\n🔹 Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1_score:.4f}")

print("\n🔹 Classification Report:\n", report)

# 📌 Step 10: Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Diabetic"], yticklabels=["Normal", "Diabetic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📌 Step 11: Generate Plots
# Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm")
plt.title("Scatter Plot of Feature 1 vs Feature 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.show()

# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(sorted(y_test), label="Actual")
plt.plot(sorted(y_pred), label="Predicted", linestyle="dashed")
plt.title("Line Plot of Predictions")
plt.legend()
plt.grid()
plt.show()

# Histogram
plt.figure(figsize=(6, 4))
plt.hist(X_train[:, 0], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram of Feature 1")
plt.xlabel("Feature 1")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
labels = ["Normal", "Diabetic"]
values = [sum(y_train == 0), sum(y_train == 1)]
plt.bar(labels, values, color=["green", "red"])
plt.title("Bar Chart of Class Distribution")
plt.ylabel("Count")
plt.grid()
plt.show()
