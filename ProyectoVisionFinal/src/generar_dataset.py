import cv2
import os
import numpy as np

# 📍 Ruta absoluta al proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 📂 dataset/Camilo/sin_procesar
RUTA_DATASET = os.path.join(BASE_DIR, "dataset")

IMAGENES_POR_FOTO = 20

for persona in os.listdir(RUTA_DATASET):
    ruta_persona = os.path.join(RUTA_DATASET, persona)

    if not os.path.isdir(ruta_persona):
        continue

    ruta_entrada = os.path.join(ruta_persona, "sin_procesar")
    ruta_salida = os.path.join(ruta_persona, "procesadas")

    if not os.path.exists(ruta_entrada):
        continue

    os.makedirs(ruta_salida, exist_ok=True)

    contador = len(os.listdir(ruta_salida))

    for img_nombre in os.listdir(ruta_entrada):
        ruta_img = os.path.join(ruta_entrada, img_nombre)
        imagen = cv2.imread(ruta_img)

        if imagen is None:
            continue

        for _ in range(IMAGENES_POR_FOTO):
            filas, cols, _ = imagen.shape

            # 🔄 Rotación leve
            angulo = np.random.randint(-10, 10)
            M = cv2.getRotationMatrix2D((cols / 2, filas / 2), angulo, 1)
            rotada = cv2.warpAffine(imagen, M, (cols, filas))

            # 💡 Variación de brillo
            brillo = np.random.randint(-25, 25)
            final = cv2.convertScaleAbs(rotada, alpha=1, beta=brillo)

            nombre_salida = f"{persona}_{contador}.jpg"
            cv2.imwrite(os.path.join(ruta_salida, nombre_salida), final)
            contador += 1

    print(f"✅ {persona}: dataset generado correctamente")
