import tkinter as tk
from tkinter import Label, Button, Scale
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import time
from snap7.util import *
from snap7.client import Client
import threading

# Kết nối với PLC
PLC_DB_NUMBER = 3  # Data block của PLC (DB3)
PLC_IP = '192.168.0.1'  # Địa chỉ IP của PLC
PLC_RACK = 0
PLC_SLOT = 1

client = Client()
client.connect(PLC_IP, PLC_RACK, PLC_SLOT)


def send_bool_signal(bit_offset, value, reset_delay=200):
    """Gửi tín hiệu Bool tới PLC và tự động đặt lại về False sau thời gian reset_delay (ms)."""
    byte_offset = bit_offset // 8  # Xác định byte nào chứa bit
    bit_within_byte = bit_offset % 8  # Xác định bit nào trong byte
    data = client.db_read(PLC_DB_NUMBER, byte_offset, 1)  # Đọc byte tương ứng từ DB

    set_bool(data, 0, bit_within_byte, value)  # Ghi giá trị vào bit
    client.db_write(PLC_DB_NUMBER, byte_offset, data)  # Ghi byte lại vào PLC

    if value:
        root.after(reset_delay, lambda: send_bool_signal(bit_offset, False))


def send_real_signal(real_offset, value):
    """Gửi tín hiệu Real tới PLC."""
    data = client.db_read(PLC_DB_NUMBER, real_offset, 4)
    set_real(data, 0, value)
    client.db_write(PLC_DB_NUMBER, real_offset, data)

def send_start_signal():
    send_bool_signal(0, True)  # SMU START (Offset 0.0)

def send_stop_signal():
    send_bool_signal(1, True)  # SMU STOP (Offset 0.1)

def send_reset_signal():
    send_bool_signal(2, True)  # SMU RESET (Offset 0.2)
    send_real_signal(2, 0)  # Đặt tốc độ về 0

def change_speed(value):
    """Gửi giá trị tốc độ tới PLC."""
    try:
        speed = float(value)
        send_real_signal(2, speed)  # SMU SPEED (Offset 2.0)
    except ValueError:
        print("Giá trị tốc độ không hợp lệ!")

# Hàm gửi tín hiệu True và sau đó là False
def send_incorrect_signal():
    """Gửi tín hiệu Bool tới PLC khi phát hiện nắp chai không chính xác."""
    send_bool_signal(80, True)  # Offset 10.0 tương đương với bit thứ 80 (10 * 8 + 0)


# Load the model and class names
model = load_model("saved_model_dir")
class_names = open("labels.txt", "r").readlines()

# Set confidence threshold
confidence_threshold = 0.8
stable_prediction = None
last_sent_time = 0
send_interval = 3  # Minimum interval between sending results (seconds)

# Thêm bộ đệm cho dự đoán
prediction_buffer = []

# Số lượng khung hình để tính trung bình
buffer_size = 5


def predict_class_with_buffer(frame, model):
    global prediction_buffer

    try:
        # Resize và chuẩn hóa ảnh
        processed_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        processed_frame = np.asarray(processed_frame, dtype=np.float32).reshape(1, 224, 224, 3)
        processed_frame = (processed_frame / 127.5) - 1

        # Dự đoán với mô hình
        prediction = model.predict(processed_frame)
        prediction_buffer.append(prediction[0])

        # Nếu chưa đủ khung hình, đợi thêm
        if len(prediction_buffer) < buffer_size:
            return None, None

        # Nếu đủ khung hình, tính trung bình dự đoán
        avg_prediction = np.mean(prediction_buffer, axis=0)
        index = np.argmax(avg_prediction)
        confidence_score = avg_prediction[index]

        # Xóa phần tử đầu tiên (FIFO)
        prediction_buffer.pop(0)

        if confidence_score >= confidence_threshold:
            return index, confidence_score
        return None, None
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None


def update_frame(camera, app):
    running = True

    while running:
        ret, frame = camera.read()
        if not ret:
            print("Không nhận được khung hình từ camera!")
            break

        # Hiển thị khung hình trên giao diện GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((app.canvas.winfo_width(), app.canvas.winfo_height()), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        app.canvas.imgtk = imgtk
        app.canvas.configure(image=imgtk)

        # Dự đoán lớp và xử lý kết quả
        predicted_class, confidence_score = predict_class_with_buffer(frame, model)
        if predicted_class is not None:
            # Hiển thị kết quả
            if predicted_class == 2:  # Nắp đóng chặt
                app.result_label.config(text="CORRECT CAP", fg="green")
                app.good_products += 1
                app.total_products += 1
                print("Correct Cap")
                time.sleep(1.5)
            elif predicted_class == 3:  # Nắp không chặt
                app.result_label.config(text="INCORRECT CAP", fg="orange")
                app.error_products += 1
                app.total_products += 1
                send_incorrect_signal()
                print("Loose Cap")
                time.sleep(1.5)
            elif predicted_class == 0:  # Không nắp
                app.result_label.config(text="NO CAP", fg="red")
                app.error_products += 1
                app.total_products += 1
                send_incorrect_signal()
                print("No Cap")
                time.sleep(1.5)
            elif predicted_class == 1:  # Không có gì
                app.result_label.config(text="NOTHING", fg="blue")
                print("No Object Detected")

            # Cập nhật thông tin sản phẩm
            app.total_label.config(text=f"Total: {app.total_products}")
            app.good_label.config(text=f"Good: {app.good_products}")
            app.error_label.config(text=f"Error: {app.error_products}")
time.sleep(0.5)


class GUIApp:
    def __init__(self, root):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Không thể mở camera!")
            exit()

        # GUI components
        self.root = root
        self.root.title("Hệ Thống Kiểm Tra Nắp Chai")
        self.root.geometry("1200x800")
        self.root.configure(bg="white")
        self.root.state('zoomed')

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=1)

        left_frame = tk.Frame(self.root, bg="white")
        left_frame.grid(row=0, column=0, sticky="nswe")
        center_frame = tk.Frame(self.root, bg="white")
        center_frame.grid(row=0, column=1, sticky="nswe")
        right_frame = tk.Frame(self.root, bg="white")
        right_frame.grid(row=0, column=2, sticky="nswe")

        # Left Frame
        self.left_image = Image.open("logo-hust.png")
        self.left_image = self.left_image.resize((150, 150), Image.Resampling.LANCZOS)
        self.left_photo = ImageTk.PhotoImage(self.left_image)
        Label(left_frame, image=self.left_photo, bg="white").pack(pady=10)

        Label(left_frame, text="CONTROL", bg="white", fg="green", font=("Arial", 20, "bold")).pack(pady=10)

        button_width = 15
        button_height = 3

        button_frame = tk.Frame(left_frame, bg="white")
        button_frame.pack(pady=10)

        # Canvas bên cạnh nút
        self.canvas_left = tk.Canvas(button_frame, bg="white", width=200, height=250, highlightthickness=0)
        self.canvas_left.grid(row=0, column=1, rowspan=3, padx=5)

        # Tạo các hình tròn
        self.circles = [
            (self.canvas_left.create_oval(20, 20, 80, 80, fill="lightgrey", outline="black"), "lightgrey"),
            (self.canvas_left.create_oval(20, 90, 80, 150, fill="lightgrey", outline="black"), "lightgrey"),
            (self.canvas_left.create_oval(20, 160, 80, 220, fill="lightgrey", outline="black"), "lightgrey")
        ]

        # Các nút điều khiển
        Button(button_frame, text="START", bg="green", fg="white", font=("Arial", 16), width=button_width, height=button_height,
               command=self.send_start_signal).grid(row=0, column=0, pady=5)
        Button(button_frame, text="STOP", bg="red", fg="white", font=("Arial", 16), width=button_width, height=button_height,
               command=self.send_stop_signal).grid(row=1, column=0, pady=5)
        Button(button_frame, text="RESET", bg="yellow", fg="black", font=("Arial", 16), width=button_width, height=button_height,
               command=self.send_reset_signal).grid(row=2, column=0, pady=5)

        Label(left_frame, text="Speed Control", bg="white", font=("Arial", 16, "bold")).pack(pady=5)
        self.speed_slider = Scale(left_frame, from_=0, to=4000, orient=tk.HORIZONTAL, length=350, bg="white", font=("Arial", 14), command=lambda v: self.change_speed(v))
        self.speed_slider.set(2000)  # Đặt giá trị ban đầu là 2000
        self.speed_slider.pack()

        # Center Frame
        Label(center_frame, text="ĐỒ ÁN TỐT NGHIỆP", bg="white", fg="red", font=("Arial", 50, "bold")).pack(pady=30)

        self.canvas = Label(center_frame, bg="black", width=800, height=550)
        self.canvas.pack(expand=True)

        result_frame = tk.Frame(center_frame, bg="white", bd=2, relief="solid")
        result_frame.pack(pady=10)
        self.result_label = Label(result_frame, text="RESULT", bg="white", fg="blue", font=("Arial", 30, "bold"), width=40, height=5)
        self.result_label.pack()
        # Right Frame
        self.right_image = Image.open("logo2.jpg")
        self.right_image = self.right_image.resize((200, 200), Image.Resampling.LANCZOS)
        self.right_photo = ImageTk.PhotoImage(self.right_image)
        Label(right_frame, image=self.right_photo, bg="white").pack(pady=10)

        Label(right_frame, text="STATUS", bg="white", fg="green", font=("Arial", 20, "bold")).pack(pady=10)

        self.total_products = 0
        self.good_products = 0
        self.error_products = 0

        self.total_label = Label(right_frame, text=f"Total: {self.total_products}", bg="white", font=("Arial", 16))
        self.total_label.pack()
        self.good_label = Label(right_frame, text=f"Good: {self.good_products}", bg="white", font=("Arial", 16))
        self.good_label.pack()
        self.error_label = Label(right_frame, text=f"Error: {self.error_products}", bg="white", font=("Arial", 16))
        self.error_label.pack()

        info_frame = tk.Frame(right_frame, bg="white")
        info_frame.pack(pady=(40, 10))  # Tăng padding trên để dịch xuống
        Label(info_frame, text="INFORMATION", bg="white", fg="blue", font=("Arial", 16)).pack(pady=5)
        Label(info_frame, text="GVHD:", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="TS. Dương Văn Lạc", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="Sinh viên thực hiện:", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="Nguyễn Việt Hoàng", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="20195026", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text=" Lê Quang Hiếu", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="20205309", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="Nguyễn Xuân Quang", bg="white", font=("Arial", 14)).pack()
        Label(info_frame, text="20184591", bg="white", font=("Arial", 14)).pack()

    def highlight_circle(self, circle_id, color):
        for cid, default_color in self.circles:
            self.canvas_left.itemconfig(cid, fill=default_color)
        self.canvas_left.itemconfig(circle_id, fill=color)

    def send_start_signal(self):
        self.highlight_circle(self.circles[0][0], "green")
        send_bool_signal(0, True)

    def send_stop_signal(self):
        self.highlight_circle(self.circles[1][0], "red")
        send_bool_signal(1, True)

    def send_reset_signal(self):
        self.highlight_circle(self.circles[2][0], "yellow")
        send_bool_signal(2, True)
        self.total_products = 0
        self.good_products = 0
        self.error_products = 0
        self.result_label.config(text="Cap Result: RESET")
        self.total_label.config(text=f"Total: {self.total_products}")
        self.good_label.config(text=f"Good: {self.good_products}")
        self.error_label.config(text=f"Error: {self.error_products}")
        print("Đã reset lại hệ thống")

    def change_speed(self, value):
        try:
            speed = int(value)
            send_real_signal(2, speed)
        except ValueError as e:
            print(f"Lỗi khi thay đổi tốc độ: {e}")

    def on_closing(self):
        self.camera.release()
        self.root.destroy()

root = tk.Tk()
app = GUIApp(root)
threading.Thread(target=update_frame, args=(app.camera, app), daemon=True).start()
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()