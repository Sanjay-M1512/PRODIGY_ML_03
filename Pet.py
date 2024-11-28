import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

cats_path = r"C:\Users\HP\OneDrive\Desktop\Intern\PetImages\Cat"
dogs_path = r"C:\Users\HP\OneDrive\Desktop\Intern\PetImages\Dog"

img_size = (128, 128)  

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                images.append(img_resized)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return images, labels

def extract_hog_features(images):
    hog_features = []
    for image in tqdm(images, desc="Extracting HOG features"):
        features = hog(image,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       visualize=False)
        hog_features.append(features)
    return np.array(hog_features)


print("Loading dataset...")
cat_images, cat_labels = load_images_from_folder(cats_path, label=0)
dog_images, dog_labels = load_images_from_folder(dogs_path, label=1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Extracting HOG features for training data...")
X_train_hog = extract_hog_features(X_train)
print("Extracting HOG features for testing data...")
X_test_hog = extract_hog_features(X_test)

scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

print("Training SVM...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_hog, y_train)
print("SVM training complete.")

print("Evaluating model...")
y_pred = svm.predict(X_test_hog)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

