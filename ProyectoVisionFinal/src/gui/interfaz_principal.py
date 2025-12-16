import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

class VisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento - Proyecto Final")
        self.root.geometry("1100x650")
        self.root.configure(bg="#1e1e1e")

        self.camera_index = 0
        self.cap = None
        self.running = False

        self.build_interface()

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()
    
    # ============================
    #  UI COMPLETA
    # ============================
    def build_interface(self):

        # === MARCO DE VIDEO ===
        self.video_frame = tk.Label(self.root, bg="#000")
        self.video_frame.place(x=20, y=20, width=780, height=560)

        # === PANEL LATERAL ===
        side_panel = tk.Frame(self.root, bg="#252526")
        side_panel.place(x=820, y=20, width=260, height=560)

        title = tk.Label(side_panel, text="Controles", fg="white", bg="#252526",
                         font=("Segoe UI", 16, "bold"))
        title.pack(pady=20)

        # Botones principales
        style = ttk.Style()
        style.configure("TButton",
            font=("Segoe UI", 12),
            padding=10
        )

        btn1 = ttk.Button(side_panel, text="Iniciar Cámara",
                          command=self.start_camera)
        btn1.pack(fill="x", padx=20, pady=10)

        btn_stop = ttk.Button(side_panel, text="Detener Cámara",
                              command=self.stop_camera)
        btn_stop.pack(fill="x", padx=20, pady=10)

        btn2 = ttk.Button(side_panel, text="Reconocimiento Facial",
                          command=self.run_face_recognition)
        btn2.pack(fill="x", padx=20, pady=10)

        btn3 = ttk.Button(side_panel, text="Reconocimiento de Objetos",
                          command=self.run_object_detection)
        btn3.pack(fill="x", padx=20, pady=10)

        btn4 = ttk.Button(side_panel, text="Guardar Captura",
                          command=self.save_frame)
        btn4.pack(fill="x", padx=20, pady=10)

        btn5 = ttk.Button(side_panel, text="Configurar Cámara",
                          command=self.change_camera)
        btn5.pack(fill="x", padx=20, pady=10)

        # === MENÚ SUPERIOR ===
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Salir", command=self.root.quit)
        menu.add_cascade(label="Archivo", menu=file_menu)

    # ============================
    #  MANEJO DE CÁMARA
    # ============================
    def start_camera(self):
        if self.running:
            return
        
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (780, 560))

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_frame.imgtk = img
            self.video_frame.config(image=img)
        
        self.root.after(10, self.update_frame)

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_frame.config(image='')

    # ============================
    #  PLACEHOLDERS PARA MÓDULOS
    # ============================
    def run_face_recognition(self):
        messagebox.showinfo("Reconocimiento Facial", "Aquí se ejecutará tu modelo facial.")

    def run_object_detection(self):
        messagebox.showinfo("Detección de Objetos", "Aquí irá YOLO, SSD, o tu modelo personalizado.")

    def save_frame(self):
        messagebox.showinfo("Guardar", "Función de guardar imagen aún no implementada.")

    def change_camera(self):
        self.camera_index = (self.camera_index + 1) % 3
        messagebox.showinfo("Cámara", f"Cambiado a cámara {self.camera_index}")

# ============================
#  MAIN
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = VisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
