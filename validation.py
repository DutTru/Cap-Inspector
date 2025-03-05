from sklearn.model_selection import train_test_split
import os
import shutil

# Đường dẫn đến thư mục dữ liệu gốc
data_dir = "datas"
train_dir = "datas_split/train"
val_dir = "datas_split/val"
test_dir = "datas_split/test"

# Tạo các thư mục train, val và test
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Chia dữ liệu theo từng lớp
for label in os.listdir(data_dir):  # Duyệt qua từng lớp (correct, incorrect)
    label_dir = os.path.join(data_dir, label)
    if not os.path.isdir(label_dir):
        continue

    # Lấy danh sách ảnh
    images = os.listdir(label_dir)

    # Chia dữ liệu thành 80% train và 20% (val + test)
    train_images, val_test_images = train_test_split(images, test_size=0.2, random_state=30)

    # Chia tiếp val_test_images thành 50% validation và 50% test
    val_images, test_images = train_test_split(val_test_images, test_size=0.5, random_state=30)

    # Tạo thư mục con cho lớp trong train, val và test
    label_train_dir = os.path.join(train_dir, label)
    label_val_dir = os.path.join(val_dir, label)
    label_test_dir = os.path.join(test_dir, label)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    # Copy ảnh vào thư mục train
    for img in train_images:
        shutil.copy(os.path.join(label_dir, img), label_train_dir)

    # Copy ảnh vào thư mục val
    for img in val_images:
        shutil.copy(os.path.join(label_dir, img), label_val_dir)

    # Copy ảnh vào thư mục test
    for img in test_images:
        shutil.copy(os.path.join(label_dir, img), label_test_dir)
