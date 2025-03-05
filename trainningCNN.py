import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def prepare_data(base_path, img_size=(224, 224), validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=validation_split)

    train_generator = datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        base_path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(base_path, output_model_path):
    train_gen, val_gen = prepare_data(base_path)

    input_shape = (224, 224, 3)
    num_classes = len(train_gen.class_indices)
    model = build_model(input_shape, num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=6
    )

    model.save(output_model_path)
    print(f"Mô hình đã được lưu tại {output_model_path}")

if __name__ == "__main__":
    base_path = "data"
    output_model_path = "model1.h5"
    train_model(base_path, output_model_path)
