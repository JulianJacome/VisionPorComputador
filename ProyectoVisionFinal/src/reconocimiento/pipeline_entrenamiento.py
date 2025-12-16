import cv2
import os
import numpy as np

def entrenar_modelo():
    # Rutas a las carpetas de datos
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    camilo_dir = os.path.join(data_dir, 'Camilo', 'procesadas')
    julian_dir = os.path.join(data_dir, 'Julian', 'procesadas')

    # Crear el reconocedor
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    ids = []

    # Función para procesar imágenes de una persona
    def process_images(person_dir, person_id):
        if not os.path.exists(person_dir):
            print(f"Directorio {person_dir} no existe.")
            return
        for file in os.listdir(person_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(person_dir, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in face_rects:
                    face = gray[y:y+h, x:x+w]
                    faces.append(face)
                    ids.append(person_id)

    # Procesar imágenes de Camilo (id=0) y Julian (id=1)
    process_images(camilo_dir, 0)
    process_images(julian_dir, 1)

    if len(faces) == 0:
        print("No se encontraron rostros en las imágenes.")
        return

    # Entrenar el modelo
    recognizer.train(faces, np.array(ids))

    # Guardar el modelo
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'modelos', 'face_recognizer.yml')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    recognizer.save(model_path)
    print(f"Modelo entrenado y guardado en {model_path}")

if __name__ == "__main__":
    entrenar_modelo()
