import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load model và tên lớp
model = load_model("saved_model_dir")
class_names = open("labels.txt", "r").readlines()

# Ngưỡng độ tin cậy
confidence_threshold = 0.8

# Bộ đệm dự đoán
prediction_buffer = []
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

# Kết nối với camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Real-time Object Detection")

# Label hiển thị khung hình từ camera
video_label = Label(root)
video_label.pack()

# Label hiển thị kết quả nhận diện
result_label = Label(root, text="Waiting for camera...", font=("Arial", 14))
result_label.pack()

# Nút thoát
exit_button = Button(root, text="Exit", command=root.quit)
exit_button.pack()

def update_frame():
    ret, frame = cap.read()
    if ret:
        # Chuyển đổi màu từ BGR (OpenCV) sang RGB (PIL)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Nhận diện từ frame
        index, confidence = predict_class_with_buffer(frame, model)
        if index is not None:
            result_text = f"Detected: {class_names[index].strip()} ({confidence*100:.1f}%)"
        else:
            result_text = "No confident prediction yet."
        result_label.config(text=result_text)

        # Chuyển frame sang hình ảnh PIL và cập nhật vào label
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Giữ tham chiếu để không bị xóa
        video_label.configure(image=imgtk)
    # Cập nhật lại sau 10ms
    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Bắt đầu vòng lặp cập nhật khung hình
update_frame()
root.mainloop()
