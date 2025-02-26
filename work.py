import numpy as np
import os
import sys
import cv2
import face_recognition
import math
import requests
import struct
import time

# API Endpoint
API_URL = "http://127.0.0.1:8000/api/users/"

# Define Struct Format
USER_STRUCT_FORMAT = "I 20s 20s 50s 15s 50s 20s 20s f f"

# Fetch user data from API
response = requests.get(API_URL)
if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit()
users = response.json()

# Store users in a dictionary (image filename -> user details)
user_dict = {}
save_path = "/Users/manishmaharjan/Downloads/face/faces"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for user in users:
    image_url = "http://127.0.0.1:8000" + user.get("image")
    image_filename = image_url.split("/")[-1]
    user_dict[image_filename] = user  # Store user details using image filename

    img_response = requests.get(image_url)
    image_filepath = os.path.join(save_path, image_filename)
    with open(image_filepath, "wb") as file:
        file.write(img_response.content)
    
    print(f"✅ Image downloaded: {image_filename}")

# Helper function to calculate confidence
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 2.0))) * 100
        return str(round(value, 2)) + "%"

# Face Recognition Class
class FaceRecognition:
    def __init__(self, price_deduction=2.0):
        self.known_face_encodings = []
        self.known_face_names = []
        self.image_ids = []  # Store image filenames (for ID matching)

        self.face_timers = {}  # Dictionary to track face entry times
        self.price_deduction = price_deduction
        self.user_balances = {}  # Store user balances
        
        self.encode_faces()
        
    
    def encode_faces(self):
        # Load known faces from folder
        for image in os.listdir("faces"):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split(".")[0])  # Use filename as name
            self.image_ids.append(image)
            # Initialize balance (e.g., starting with $10)
            self.user_balances[image.split(".")[0]] = 10.0

    def run_recognition(self):
        """Runs real-time face recognition."""
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Error: Camera not found...")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            detected_faces = []

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                name = "Unknown"
                fullname = "Unknown"
                confidence = "Unknown"

                stopA = "Unknown"
                stopB = "Unknown"
                
                if matches[best_match_index]:
                    matched_image = self.image_ids[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])
                    name = self.known_face_names[best_match_index]
                    detected_faces.append(name)
                    if matched_image in user_dict:
                        user = user_dict[matched_image]
                        fullname = f"{user['first_name']} {user['last_name']}"
                        stopA = user["source"]


                    # If face enters for the first time, start timer
                    if name not in self.face_timers:
                        self.face_timers[name] = time.time()
                        print(f"{fullname} entered at {time.ctime(self.face_timers[name])}")
                    else:
                        elapsed_time = time.time() - self.face_timers[name]
                        if elapsed_time >= 5:  # After 5 seconds, set stopB
                            stopB = "Grafton"
                    


                                            
                 
                        

                # Draw rectangle and label
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)
                cv2.putText(frame, f"{fullname} ({confidence})", (left + 6, bottom - 6), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"Balance: {self.user_balances.get(name, 0.0):.2f}", (left + 6, bottom + 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"Stop A: {stopA}", (left + 6, bottom + 64), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"Stop B: {stopB}", (left + 6, bottom + 93), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
       
            
            
            # Check for faces that have left
            for stored_name in list(self.face_timers.keys()):
                if stored_name not in detected_faces:  # If face is no longer detected
                    elapsed_time = time.time() - self.face_timers[stored_name]
                    if elapsed_time > 2:  # If face is gone for more than 2 seconds
                        # Retrieve user's full name from user_dict
                        matched_image = f"{stored_name}.jpg"  # Adjust extension if necessary
                        fullname = user_dict.get(matched_image, {}).get('first_name', 'Unknown') + " " + user_dict.get(matched_image, {}).get('last_name', 'Unknown')

                        if stored_name in self.user_balances:
                            self.user_balances[stored_name] -= self.price_deduction  # Deduct balance
                            print(f"{fullname} left. Deducted ${self.price_deduction}. New balance: ${self.user_balances[stored_name]:.2f}")
            
                        del self.face_timers[stored_name]  # Remove from tracking
            
            
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Run Face Recognition
if __name__ == "__main__":
    fr = FaceRecognition(price_deduction=2.0)
    fr.run_recognition()
