import cv2
import numpy as np
from pathlib import Path

# Ruta base del dataset
BASE = Path(__file__).parent.parent.parent / "dataset"

# Detector frontal de OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                      "haarcascade_frontalface_default.xml")

# ---------------------------
# Procesar y segmentar imágenes
# ---------------------------
def procesar_persona(nombre):
    folder_raw = BASE / nombre / "sin_procesar"
    folder_proc = BASE / nombre / "procesadas"
    folder_seg = BASE / nombre / "segmentadas"

    folder_proc.mkdir(parents=True, exist_ok=True)
    folder_seg.mkdir(parents=True, exist_ok=True)

    for file in folder_raw.iterdir():
        if not file.is_file() or file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        path = str(file.resolve())

        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # ---------------------------
        # PROCESADAS (recorte + resize)
        # ---------------------------
        img_resized = cv2.resize(img, (300, 300))
        proc_path = str(folder_proc / file.name)
        success, encoded = cv2.imencode('.jpg', img_resized)
        if success:
            with open(proc_path, 'wb') as f:
                f.write(encoded.tobytes())

        # ---------------------------
        # SEGMENTADAS (solo el rostro)
        # ---------------------------
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            rostro = img[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200))
            seg_path = str(folder_seg / file.name)
            success, encoded = cv2.imencode('.jpg', rostro)
            if success:
                with open(seg_path, 'wb') as f:
                    f.write(encoded.tobytes())
        else:
            # Si no detecta rostro, igual guarda algo para no perder imagen
            seg_path = str(folder_seg / ("no_rostro_" + file.name))
            success, encoded = cv2.imencode('.jpg', img)
            if success:
                with open(seg_path, 'wb') as f:
                    f.write(encoded.tobytes())

    print(f"[OK] Procesado completo para: {nombre}")


# ---------------------------
# Ejecutar para cada persona
# ---------------------------
def main():
    personas = ["Julian", "Camilo"]  # <-- Cambiar si agregan más personas

    for p in personas:
        procesar_persona(p)

if __name__ == "__main__":
    main()
