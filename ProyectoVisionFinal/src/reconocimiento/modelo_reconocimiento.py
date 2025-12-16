import cv2
import os

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.names = {0: "Camilo", 1: "Julian"}
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos', 'face_recognizer.yml')
        if os.path.exists(model_path):
            self.recognizer.read(model_path)
            self.model_loaded = True
            print("Modelo cargado.")
        else:
            print("Modelo no encontrado. Entrena el modelo primero.")
            self.model_loaded = False

    def recognize_face(self, frame):
        if not self.model_loaded:
            return []  # No reconocer si no hay modelo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        recognized_faces = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            id_, confidence = self.recognizer.predict(face)
            if confidence < 100:  # Umbral de confianza
                name = self.names.get(id_, "Desconocido")
            else:
                name = "Desconocido"
            recognized_faces.append((x, y, w, h, name, confidence))
        return recognized_faces
