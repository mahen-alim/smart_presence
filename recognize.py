import cv2
import mysql.connector
import numpy as np
import os
from tensorflow.keras.models import load_model

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="face_recognition_db"
    )

def fetch_face_images():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT name, face_image FROM users")
    users = cursor.fetchall()
    cursor.close()
    db.close()
    return [(name, cv2.imdecode(np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR)) for name, face_image in users]

def load_face_recognition_model(model_path="model.keras"):
    model = load_model(model_path)
    return model

def preprocess_face_image(face_image, target_size=(128, 128)):
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
    elif face_image.shape[2] != 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    face_image = cv2.resize(face_image, target_size)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def extract_face_features(model, face_image):
    preprocessed_face = preprocess_face_image(face_image)
    features = model.predict(preprocessed_face)
    return features

def recognize_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = load_face_recognition_model()

    users = fetch_face_images()
    if not users:
        print("No face images found in database.")
        return

    user_features = [(name, extract_face_features(model, face_image)) for name, face_image in users]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_features = extract_face_features(model, face_image)

            best_match_name = "Unknown"
            best_match_score = float('inf')

            for name, user_feature in user_features:
                score = np.linalg.norm(face_features - user_feature)
                if score < best_match_score:
                    best_match_name = name
                    best_match_score = score

            threshold = 0.5
            if best_match_score > threshold:
                best_match_name = "Unknown"
            print(f"Detected: {best_match_name}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, best_match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
