import cv2
import os

# Khởi tạo thư mục lưu dữ liệu
data_folders = ["datas/Nap_Dong_Chat", "datas/Khong_Nap", "datas/Nap_Khong_Chat", "datas/Khong_Co_Gi"]
for folder in data_folders:
    os.makedirs(folder, exist_ok=True)

# Khởi tạo bộ đếm số ảnh
valid_extensions = ('.jpg', '.jpeg', '.png')
data_counters = {
    folder: len([f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)])
    for folder in data_folders
}

# Khởi tạo camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra kết nối.")
    exit()

print("Hướng dẫn:")
print("  - Nhấn '0' để lưu ảnh 'Nap_Dung'")
print("  - Nhấn '1' để lưu ảnh 'Khong_Nap'")
print("  - Nhấn '2' để lưu ảnh 'Nap_Lech'")
print("  - Nhấn '3' để lưu ảnh 'Khong_Co_Gi'")
print("  - Nhấn 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Data Collection", frame)

    # Chọn nhãn ảnh
    key = cv2.waitKey(1)
    if key == ord('0'):
        folder = data_folders[0]
    elif key == ord('1'):
        folder = data_folders[1]
    elif key == ord('2'):
        folder = data_folders[2]
    elif key == ord('3'):
        folder = data_folders[3]
    elif key == ord('q'):
        break
    else:
        continue

    # Lưu ảnh với tên được đánh số tự động
    data_counters[folder] += 1
    filename = os.path.join(folder, f"image_{data_counters[folder]}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Lưu ảnh: {filename}")

cap.release()
cv2.destroyAllWindows()
