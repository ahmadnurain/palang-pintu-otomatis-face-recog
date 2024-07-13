import cv2
import os
import pickle
import face_recognition
from sklearn.neighbors import KNeighborsClassifier

dataset_path = 'dataset'
knn_model_path = 'knn_model.pkl'
id_to_name_path = 'id_to_name.pkl'

# Load trained KNN model and ID-to-name mapping
if os.path.exists(knn_model_path) and os.path.exists(id_to_name_path):
    with open(knn_model_path, 'rb') as f:
        knn_model = pickle.load(f)
    with open(id_to_name_path, 'rb') as f:
        id_to_name = pickle.load(f)
else:
    print("Model and ID-to-name mapping not found.")
    exit()

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def validate_and_remove_face(face_encoding):
    encoding_reshaped = face_encoding.reshape(1, -1)
    distances, indices = knn_model.kneighbors(encoding_reshaped, n_neighbors=1)
    closest_distance = distances[0][0]
    name_pred = knn_model.predict(encoding_reshaped)[0]
    name = id_to_name.get(name_pred, "Unknown")

    if name == "Unknown":
        print("Access denied")
        return False

    if closest_distance < 0.6:
        if name_pred in id_to_name:
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
            print("Palang Pintu Exit Tidak Terbuka!!")
            return False
    else:
        print("Access denied")
        return False

def capture_and_validate():
    validated = False
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

            if len(encodings) > 0 and not validated:
                encoding = encodings[0]
                if validate_and_remove_face(encoding):
                    print("Palang Pintu Exit Terbuka!!")
                else:
                    print("Palang Pintu Exit Tidak Terbuka!!")
                validated = True

        cv2.imshow("Frame", frame)
        if validated:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("System shutdown")
            video.release()
            cv2.destroyAllWindows()
            exit()

def main():
    print("Press 'd' to start detection.")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'):
            capture_and_validate()
            print("Press 'd' to start detection again.")
        elif key == ord('q'):
            print("System shutdown")
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
