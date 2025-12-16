import cv2
import os

# Ruta base del dataset
BASE = "dataset"

# Detector frontal de OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                      "haarcascade_frontalface_default.xml")

# ---------------------------
# Procesar y segmentar imágenes
# ---------------------------
def procesar_persona(nombre):
    folder_raw = os.path.join(BASE, nombre, "sin_procesar")
    folder_proc = os.path.join(BASE, nombre, "procesadas")
    folder_seg = os.path.join(BASE, nombre, "segmentadas")

    os.makedirs(folder_proc, exist_ok=True)
    os.makedirs(folder_seg, exist_ok=True)

    for file in os.listdir(folder_raw):
        path = os.path.join(folder_raw, file)

        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # ---------------------------
        # PROCESADAS (recorte + resize)
        # ---------------------------
        img_resized = cv2.resize(img, (300, 300))
        proc_path = os.path.join(folder_proc, file)
        cv2.imwrite(proc_path, img_resized)

        # ---------------------------
        # SEGMENTADAS (solo el rostro)
        # ---------------------------
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            rostro = img[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200))
            seg_path = os.path.join(folder_seg, file)
            cv2.imwrite(seg_path, rostro)
        else:
            # Si no detecta rostro, igual guarda algo para no perder imagen
            seg_path = os.path.join(folder_seg, "no_rostro_" + file)
            cv2.imwrite(seg_path, img)

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
