import cv2
import numpy as np
import os
import pickle
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

dataset_path = 'dataset'
knn_model_path = 'knn_model.pkl'
id_to_name_path = 'id_to_name.pkl'

# Load trained KNN model and ID-to-name mapping if available
if os.path.exists(knn_model_path) and os.path.exists(id_to_name_path):
    with open(knn_model_path, 'rb') as f:
        knn_model = pickle.load(f)
    with open(id_to_name_path, 'rb') as f:
        id_to_name = pickle.load(f)
else:
    knn_model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    id_to_name = {}

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to register new face
def register_new_face(face_img, name):
    if name not in id_to_name.values():
        current_id = len(id_to_name)
        id_to_name[current_id] = name

        # Save face image to dataset
        user_folder = os.path.join(dataset_path, name)
        os.makedirs(user_folder, exist_ok=True)
        img_count = len(os.listdir(user_folder)) + 1
        img_path = os.path.join(user_folder, f'{name}_{img_count}.jpg')
        cv2.imwrite(img_path, face_img)

        # Encode the new face and update the model
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(rgb_face)[0]
        update_knn_model(encoding, current_id)
        
        print(f"New face registered for {name}")
    else:
        print(f"Face for {name} is already registered.")

# Function to update KNN model
def update_knn_model(new_encoding, new_id):
    encodings = []
    ids = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            current_id = list(id_to_name.keys())[list(id_to_name.values()).index(folder_name)]
            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                encodings.append(encoding)
                ids.append(current_id)

    encodings.append(new_encoding)
    ids.append(new_id)
    
    knn_model.fit(encodings, ids)
    
    with open(knn_model_path, 'wb') as f:
        pickle.dump(knn_model, f)
    with open(id_to_name_path, 'wb') as f:
        pickle.dump(id_to_name, f)

# Function to validate and remove face
def validate_and_remove_face(face_encoding):
    encoding_reshaped = face_encoding.reshape(1, -1)
    distances, indices = knn_model.kneighbors(encoding_reshaped, n_neighbors=1)
    closest_distance = distances[0][0]
    name_pred = knn_model.predict(encoding_reshaped)[0]
    name = id_to_name.get(name_pred, "Unknown")
    
    if closest_distance < 0.6:
        print(f"Access granted: {name}")
        user_folder = os.path.join(dataset_path, name)
        for file_name in os.listdir(user_folder):
            os.remove(os.path.join(user_folder, file_name))
        os.rmdir(user_folder)
        del id_to_name[name_pred]
        with open(id_to_name_path, 'wb') as f:
            pickle.dump(id_to_name, f)
        return True
    else:
        print("Access denied")
        return False

def capture_and_register():
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            cv2.putText(frame, "Register mode: Press 'r' to register", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Register Face", face_img)

            if cv2.waitKey(1) & 0xFF == ord('r'):
                new_name = input("Enter name for new face: ")
                register_new_face(face_img, new_name)
                return  # Exit the loop after registering

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def capture_and_validate():
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            if len(encodings) > 0:
                encoding = encodings[0]
                if validate_and_remove_face(encoding):
                    print("Palang pintu terbuka")
                else:
                    print("Palang pintu tidak terbuka")
                return  # Exit the loop after validation

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main loop
while True:
    mode = input("Enter mode (register/exit/quit): ").strip().lower()

    if mode == "register":
        capture_and_register()
    elif mode == "exit":
        capture_and_validate()
    elif mode == "quit":
        break

video.release()
cv2.destroyAllWindows()
print("System shutdown")
