import cv2
import mysql.connector
import numpy as np
import os

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="face_recognition_db"
    )

def save_face_image(name, face_image):
    db = connect_db()
    cursor = db.cursor()
    _, buffer = cv2.imencode('.jpg', face_image)
    face_image_blob = buffer.tobytes()
    cursor.execute("INSERT INTO users (name, face_image) VALUES (%s, %s)", (name, face_image_blob))
    db.commit()
    cursor.close()
    db.close()

    # Save image to dataset folder
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    user_dir = os.path.join(dataset_dir, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    image_path = os.path.join(user_dir, f"{name}_{len(os.listdir(user_dir)) + 1}.jpg")  # Save with incremental filename
    cv2.imwrite(image_path, face_image)
    print(f"Face image saved to {image_path}")

def capture_and_save_face(name, max_photos=30):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Capturing face for", name)
    count = 0  # Initialize count for number of photos taken
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if count < max_photos:  # Check if photo count is less than max
            for (x, y, w, h) in faces:
                face_image = gray[y:y+h, x:x+w]
                save_face_image(name, face_image)
                count += 1  # Increment photo count
                if count >= max_photos:
                    print("Maximum photo limit reached. Stopping camera.")
                    break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_photos:  # Exit if 'q' is pressed or max photo limit reached
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    capture_and_save_face(user_name)
