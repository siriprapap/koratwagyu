# -*- coding: utf-8 -*-
"""vgg-16-KoratCattle.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uE2Rqpe5ViyT_b3tiEPJBdKvspCLZFwj
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install flask flask-cors pyngrok

"""โหลด VGG16 และเตรียมโมเดล"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# โหลด VGG16 โดยไม่รวม top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# แช่แข็ง (Freeze) เลเยอร์ของ VGG16
for layer in base_model.layers:
    layer.trainable = False

# สร้างเลเยอร์ใหม่สำหรับการทำนาย
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
output = Dense(1)(x)  # สำหรับค่าน้ำหนักของโค

# สร้างโมเดลสุดท้าย
model = Model(inputs=base_model.input, outputs=output)

# คอมไพล์โมเดล
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae', 'mse', 'accuracy'])

"""การเตรียม Image Data Generator"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# สร้าง ImageDataGenerator สำหรับการเตรียมข้อมูลและการปรับขนาด
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# สร้าง generator สำหรับข้อมูลฝึก
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/BeefKorat/training',  # โฟลเดอร์ที่เก็บภาพฝึก
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Changed to 'sparse' or 'raw' for regression
    subset='training'  # ใช้ข้อมูลสำหรับการฝึก
)

# สร้าง generator สำหรับข้อมูล validation
validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/BeefKorat/validation',  # ใช้โฟลเดอร์เดียวกัน
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',  # Changed to 'sparse' or 'raw' for regression
    subset='validation'  # ใช้ข้อมูลสำหรับการทดสอบ
)

"""การฝึกโมเดล"""

train_dir = '/content/drive/MyDrive/BeefKorat/training'
validation_dir = '/content/drive/MyDrive/BeefKorat/validation'

"""#100"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Custom R-squared metric
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))  # Return R-squared value

# Load VGG16 model and unfreeze some layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output (sigmoid activation)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(),
        Recall(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),  # Added MAPE
        r_squared  # Added custom R² metric
    ]
)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks to assist training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy, precision, recall, mae, mse, mape, r_squared_value = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r_squared_value:.4f}")

result = model.evaluate(validation_generator)
print(result)

result = model.evaluate(validation_generator)
print(result)

"""##200"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Custom R-squared metric
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))  # Return R-squared value

# Load VGG16 model and unfreeze some layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output (sigmoid activation)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(),
        Recall(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),  # Added MAPE
        r_squared  # Added custom R² metric
    ]
)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks to assist training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=200,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy, precision, recall, mae, mse, mape, r_squared_value = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r_squared_value:.4f}")

result = model.evaluate(validation_generator)
print(result)

"""##300"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Custom R-squared metric
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))  # Return R-squared value

# Load VGG16 model and unfreeze some layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output (sigmoid activation)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(),
        Recall(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),  # Added MAPE
        r_squared  # Added custom R² metric
    ]
)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks to assist training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=300,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy, precision, recall, mae, mse, mape, r_squared_value = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r_squared_value:.4f}")

result = model.evaluate(validation_generator)
print(result)

"""##400"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Custom R-squared metric
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))  # Return R-squared value

# Load VGG16 model and unfreeze some layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output (sigmoid activation)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(),
        Recall(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),  # Added MAPE
        r_squared  # Added custom R² metric
    ]
)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks to assist training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=400,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy, precision, recall, mae, mse, mape, r_squared_value = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r_squared_value:.4f}")

result = model.evaluate(validation_generator)
print(result)

"""##500"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Custom R-squared metric
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))  # Total sum of squares
    return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))  # Return R-squared value

# Load VGG16 model and unfreeze some layers for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output (sigmoid activation)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(),
        Recall(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),  # Added MAPE
        r_squared  # Added custom R² metric
    ]
)

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks to assist training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=500,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy, precision, recall, mae, mse, mape, r_squared_value = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
      f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, R²: {r_squared_value:.4f}")

result = model.evaluate(validation_generator)
print(result)

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer parameters.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        """
        Update the weights using Adam optimization.

        :param weights: Current weights of the model (numpy array)
        :param gradients: Gradients computed from the loss function (numpy array)
        :return: Updated weights
        """
        if self.m is None:  # Initialize moment vectors
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients  # Update biased first moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)  # Update biased second moment

        # Correct bias for moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        updated_weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weights


# Example usage
if __name__ == "__main__":
    optimizer = AdamOptimizer(learning_rate=0.001)
    weights = np.array([0.5, -0.3, 0.8])
    gradients = np.array([0.1, -0.2, 0.05])

    new_weights = optimizer.update(weights, gradients)
    print("Updated weights with Adam:", new_weights)

import numpy as np

class SGDMomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize the SGDM optimizer with a learning rate and momentum.

        :param learning_rate: Step size for updating the weights
        :param momentum: Momentum factor to accelerate convergence
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, gradients):
        """
        Update the weights using SGDM.

        :param weights: Current weights of the model (numpy array)
        :param gradients: Gradients computed from the loss function (numpy array)
        :return: Updated weights
        """
        if self.velocity is None:  # Initialize velocity vector
            self.velocity = np.zeros_like(weights)

        # Update velocity
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients

        # Update weights
        updated_weights = weights + self.velocity
        return updated_weights


# Example usage
if __name__ == "__main__":
    # Initialize the optimizer
    optimizer = SGDMomentumOptimizer(learning_rate=0.01, momentum=0.9)

    # Example weights and gradients
    weights = np.array([0.5, -0.3, 0.8])
    gradients = np.array([0.1, -0.2, 0.05])

    # Perform updates
    for step in range(5):  # Perform multiple steps to observe changes
        weights = optimizer.update(weights, gradients)
        print(f"Step {step + 1}: Updated weights: {weights}")

"""การประเมินผลลัพธ์"""

metrics = dict(zip(model.metrics_names, result))
print(metrics)

"""สร้างกราฟแสดงผลลัพธ์การทำนาย

กราฟแบบงานวิจัย
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ตั้งค่า font
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})

# ตรวจสอบว่า history มีค่าหรือไม่
if 'history' not in globals():
    raise ValueError("ไม่พบตัวแปร history กรุณาเทรนโมเดลก่อน!")

# ดึงค่าต่างๆ ออกจาก history
loss = np.array(history.history.get('loss', []))
accuracy = np.array(history.history.get('accuracy', []))
precision = np.array(history.history.get('precision', [np.nan] * len(loss)))  # ใช้ NaN ถ้าไม่มีค่า
recall = np.array(history.history.get('recall', [np.nan] * len(loss)))

val_precision = np.array(history.history.get('val_precision', [np.nan] * len(loss)))
val_recall = np.array(history.history.get('val_recall', [np.nan] * len(loss)))

# คำนวณ F1 Score
f1_train = 2 * (precision * recall) / (precision + recall + 1e-10)
f1_val = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)

# ใช้ np.nan_to_num() ป้องกันข้อผิดพลาดจาก NaN
metrics = np.vstack([
    np.nan_to_num(loss),
    np.nan_to_num(accuracy),
    np.nan_to_num(precision),
    np.nan_to_num(recall)
])

# สร้างกราฟ
fig, axs = plt.subplots(3, 2, figsize=(22, 20))
fig.suptitle('Enhanced Visualizations of Model Metrics', fontsize=30, fontweight='bold')

# 1️⃣ Scatter Plot ของ F1 Score
axs[0, 0].scatter(range(len(f1_train)), f1_train, color='darkblue', label='Training F1', s=60)
axs[0, 0].scatter(range(len(f1_val)), f1_val, color='darkred', label='Validation F1', s=60)
axs[0, 0].set_title('Scatter Plot of F1 Score', fontsize=24)
axs[0, 0].set_xlabel('Epoch', fontsize=20)
axs[0, 0].set_ylabel('F1 Score', fontsize=20)
axs[0, 0].legend(fontsize=18)
axs[0, 0].grid(True)

# 2️⃣ Heatmap ของค่าต่างๆ
sns.heatmap(metrics, annot=True, fmt=".2f", cmap="YlGnBu", ax=axs[0, 1], cbar_kws={'label': 'Metric Value'})
axs[0, 1].set_title('Heatmap of Metrics', fontsize=24)
axs[0, 1].set_yticks([0.5, 1.5, 2.5, 3.5])
axs[0, 1].set_yticklabels(['Loss', 'Accuracy', 'Precision', 'Recall'], fontsize=16)

# 3️⃣ Histogram ของ Accuracy
axs[1, 0].hist(accuracy, bins=10, color='darkgreen', alpha=0.7, label='Training Accuracy')
axs[1, 0].hist(history.history.get('val_accuracy', []), bins=10, color='orange', alpha=0.7, label='Validation Accuracy')
axs[1, 0].set_title('Histogram of Accuracy', fontsize=24)
axs[1, 0].set_xlabel('Accuracy', fontsize=20)
axs[1, 0].set_ylabel('Frequency', fontsize=20)
axs[1, 0].legend(fontsize=18)
axs[1, 0].grid(True)

# 4️⃣ Bubble Chart - Precision
sizes_precision = 100 * (precision + 1e-2)
axs[1, 1].scatter(precision, val_precision, s=sizes_precision, alpha=0.5, color='purple')
axs[1, 1].set_title('Bubble Chart of Precision', fontsize=24)
axs[1, 1].set_xlabel('Training Precision', fontsize=20)
axs[1, 1].set_ylabel('Validation Precision', fontsize=20)
axs[1, 1].grid(True)

# 5️⃣ Bubble Chart - Recall
sizes_recall = 100 * (recall + 1e-2)
axs[2, 0].scatter(recall, val_recall, s=sizes_recall, alpha=0.5, color='blue')
axs[2, 0].set_title('Bubble Chart of Recall', fontsize=24)
axs[2, 0].set_xlabel('Training Recall', fontsize=20)
axs[2, 0].set_ylabel('Validation Recall', fontsize=20)
axs[2, 0].grid(True)

# 6️⃣ ปิดช่องว่างที่เหลือโดยเพิ่ม Subplot แสดงข้อความแทน
axs[2, 1].axis('off')
axs[2, 1].text(0.5, 0.5, 'Visualization Complete!', fontsize=24, ha='center', va='center', fontweight='bold')

# ปรับแต่งรูปแบบเส้นกราฟ
for ax in axs.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

# จัด layout และแสดงผล
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

import pandas as pd
import numpy as np

# ตรวจสอบว่า history มีค่าหรือไม่
if 'history' not in globals():
    raise ValueError("ไม่พบตัวแปร history กรุณาเทรนโมเดลก่อน!")

# ดึงค่าจาก history
epochs = range(1, len(history.history.get('loss', [])) + 1)

# ใช้ .get() เพื่อป้องกัน KeyError และเติม NaN ถ้าไม่มีข้อมูล
metrics_df = pd.DataFrame({
    'Epoch': epochs,
    'Training Loss': history.history.get('loss', [np.nan] * len(epochs)),
    'Validation Loss': history.history.get('val_loss', [np.nan] * len(epochs)),
    'Training Accuracy': history.history.get('accuracy', [np.nan] * len(epochs)),
    'Validation Accuracy': history.history.get('val_accuracy', [np.nan] * len(epochs)),
    'Training Precision': history.history.get('precision', [np.nan] * len(epochs)),
    'Validation Precision': history.history.get('val_precision', [np.nan] * len(epochs)),
    'Training Recall': history.history.get('recall', [np.nan] * len(epochs)),
    'Validation Recall': history.history.get('val_recall', [np.nan] * len(epochs)),
})

# คำนวณ F1 Score (ป้องกัน NaN)
metrics_df['Training F1 Score'] = 2 * (metrics_df['Training Precision'] * metrics_df['Training Recall']) / \
                                  (metrics_df['Training Precision'] + metrics_df['Training Recall'] + 1e-10)
metrics_df['Validation F1 Score'] = 2 * (metrics_df['Validation Precision'] * metrics_df['Validation Recall']) / \
                                    (metrics_df['Validation Precision'] + metrics_df['Validation Recall'] + 1e-10)

# แสดงผล 5 แถวแรก
print(metrics_df.head())

import matplotlib.pyplot as plt
import numpy as np

# ✅ ตั้งค่าฟอนต์ (รองรับภาษาไทย)
plt.rcParams.update({'font.family': 'TH Sarabun New', 'font.size': 14})

# ✅ ตรวจสอบว่า history มีค่าหรือไม่
if 'history' not in globals():
    raise ValueError("⚠️ ไม่พบตัวแปร history กรุณาเทรนโมเดลก่อน!")

# ✅ ดึงค่าจาก history (ใช้ .get() เพื่อป้องกัน KeyError)
epochs = range(len(history.history.get('loss', [])))

loss = np.array(history.history.get('loss', [np.nan] * len(epochs)))
val_loss = np.array(history.history.get('val_loss', [np.nan] * len(epochs)))
accuracy = np.array(history.history.get('accuracy', [np.nan] * len(epochs)))
val_accuracy = np.array(history.history.get('val_accuracy', [np.nan] * len(epochs)))
precision = np.array(history.history.get('precision', [np.nan] * len(epochs)))
val_precision = np.array(history.history.get('val_precision', [np.nan] * len(epochs)))
recall = np.array(history.history.get('recall', [np.nan] * len(epochs)))
val_recall = np.array(history.history.get('val_recall', [np.nan] * len(epochs)))

# ✅ คำนวณ F1 Score (ป้องกันหารด้วย 0)
f1_train = 2 * (precision * recall) / (precision + recall + 1e-10)
f1_val = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)

# ✅ เริ่มสร้างกราฟ
plt.figure(figsize=(14, 8))
plt.title('📊 Combined Line Plot of Model Metrics', fontsize=24, fontweight='bold')

# Plot Metrics
plt.plot(epochs, loss, label='Training Loss', color='red', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.plot(epochs, accuracy, label='Training Accuracy', color='blue', linewidth=2)
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='green', linewidth=2)
plt.plot(epochs, precision, label='Training Precision', color='purple', linewidth=2)
plt.plot(epochs, val_precision, label='Validation Precision', color='violet', linewidth=2)
plt.plot(epochs, recall, label='Training Recall', color='darkcyan', linewidth=2)
plt.plot(epochs, val_recall, label='Validation Recall', color='cyan', linewidth=2)
plt.plot(epochs, f1_train, label='Training F1 Score', color='darkblue', linewidth=2)
plt.plot(epochs, f1_val, label='Validation F1 Score', color='darkred', linewidth=2)

# ✅ ตั้งค่าแกนและตำนาน
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Metric Value', fontsize=16)
plt.legend(fontsize=14, loc='center right')
plt.grid(True)
plt.tight_layout()

# ✅ แสดงผล
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# ✅ ตั้งค่าฟอนต์ (รองรับภาษาไทย)
plt.rcParams.update({'font.family': 'TH Sarabun New', 'font.size': 14})

# ✅ ตรวจสอบว่า history มีค่าหรือไม่
if 'history' not in globals():
    raise ValueError("⚠️ ไม่พบตัวแปร history กรุณาเทรนโมเดลก่อน!")

# ✅ ดึงค่าจาก history (ใช้ .get() เพื่อป้องกัน KeyError)
epochs = np.arange(len(history.history.get('loss', [])))

loss = np.array(history.history.get('loss', [np.nan] * len(epochs)))
val_loss = np.array(history.history.get('val_loss', [np.nan] * len(epochs)))
accuracy = np.array(history.history.get('accuracy', [np.nan] * len(epochs)))
val_accuracy = np.array(history.history.get('val_accuracy', [np.nan] * len(epochs)))
precision = np.array(history.history.get('precision', [np.nan] * len(epochs)))
val_precision = np.array(history.history.get('val_precision', [np.nan] * len(epochs)))
recall = np.array(history.history.get('recall', [np.nan] * len(epochs)))
val_recall = np.array(history.history.get('val_recall', [np.nan] * len(epochs)))

# ✅ คำนวณ F1 Score (ป้องกันหารด้วย 0)
f1_train = 2 * (precision * recall) / (precision + recall + 1e-10)
f1_val = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)

# ✅ ปรับความกว้างของแท่ง และตำแหน่งให้ไม่ทับกัน
bar_width = 0.08

# ✅ เริ่มสร้างกราฟ
plt.figure(figsize=(18, 10))
plt.title('📊 Combined Bar Plot of Model Metrics', fontsize=24, fontweight='bold')

# Plot Training Metrics
plt.bar(epochs - 2.5 * bar_width, loss, width=bar_width, label='Training Loss', color='red')
plt.bar(epochs - 1.5 * bar_width, accuracy, width=bar_width, label='Training Accuracy', color='blue')
plt.bar(epochs - 0.5 * bar_width, precision, width=bar_width, label='Training Precision', color='purple')
plt.bar(epochs + 0.5 * bar_width, recall, width=bar_width, label='Training Recall', color='darkcyan')
plt.bar(epochs + 1.5 * bar_width, f1_train, width=bar_width, label='Training F1 Score', color='darkblue')

# Plot Validation Metrics
plt.bar(epochs - 2.5 * bar_width, val_loss, width=bar_width, label='Validation Loss', color='orange', alpha=0.7)
plt.bar(epochs - 1.5 * bar_width, val_accuracy, width=bar_width, label='Validation Accuracy', color='green', alpha=0.7)
plt.bar(epochs - 0.5 * bar_width, val_precision, width=bar_width, label='Validation Precision', color='violet', alpha=0.7)
plt.bar(epochs + 0.5 * bar_width, val_recall, width=bar_width, label='Validation Recall', color='cyan', alpha=0.7)
plt.bar(epochs + 1.5 * bar_width, f1_val, width=bar_width, label='Validation F1 Score', color='darkred', alpha=0.7)

# ✅ ตั้งค่าแกนและตำนาน
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Metric Value', fontsize=16)
plt.xticks(epochs)
plt.legend(fontsize=14, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# ✅ แสดงผล
plt.show()

"""การทดสอบโมเดล"""

from tensorflow.keras.preprocessing import image

# โหลดภาพใหม่เพื่อทดสอบ
img_path = '/content/drive/MyDrive/BeefKorat/testing/IMG_09.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ทำนายค่าน้ำหนัก
predicted_weight = model.predict(img_array)
print(f"Predicted Weight: {predicted_weight[0][0]:.2f} kg")

"""การใช้ Shaffer's formula"""

# Shaffer's formula
def shaffers_formula(length, chest_girth):
    # ค่าคงที่ a, b, c ควรหาจากข้อมูลของคุณ
    a, b, c = 0.0001, 1.5, 1.2
    return a * (length ** b) * (chest_girth ** c)

# ทำนายด้วย Shaffer's formula
length = 1.8  # ความยาว (เมตร)
chest_girth = 2.0  # เส้นรอบอก (เมตร)
estimated_weight = shaffers_formula(length, chest_girth)
print(f"Estimated Weight (Shaffer's Formula): {estimated_weight:.2f} kg")

# Example values for girth, G (weight constant), and C (correction constant)
girth = 150  # Girth in cm (ensure this value is accurate)
G = 1.5      # Adjusted weight constant for Korat Wagyu (example)
C = 1.0      # Correction constant (this is not used in this formula)

# Applying Shaffer's modified formula: (L * G)^2 / 300
weight = ((girth * G) ** 2) / 300

# Check if the calculated weight is realistic
if weight < 100:  # If the weight seems unrealistic, check inputs or constants
    print("Error: Invalid calculation, check the input values or constants.")
else:
    print(f"Estimated weight: {weight} kg")

import numpy as np
import pandas as pd

# ตัวอย่างข้อมูล
data = {
    'Measured Weight': [120, 150, 200, 180, 140],  # น้ำหนักจริงจากสายวัด
    'Predicted Weight': [115, 145, 210, 175, 135],  # น้ำหนักที่คำนวณจากสูตร
}

# สร้าง DataFrame
df = pd.DataFrame(data)

# คำนวณความคลาดเคลื่อน
df['Absolute Error'] = np.abs(df['Predicted Weight'] - df['Measured Weight'])
df['Percentage Error'] = df['Absolute Error'] / df['Measured Weight'] * 100

# คำนวณ MAE และ MAPE
mae = df['Absolute Error'].mean()
mape = df['Percentage Error'].mean()

# แสดงผล
print(df)
print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

model.save('vgg16-KoratCattle.keras')

!pip install flask-ngrok # Install the required flask_ngrok module

import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def r_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# จากนั้นโหลดโมเดลใหม่
model = tf.keras.models.load_model("/content/vgg16-KoratCattle.keras", custom_objects={"r_squared": r_squared})

model.save("/content/vgg16-KoratCattle.keras", include_optimizer=False)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# โหลดโมเดลที่ผ่านการฝึก
model = tf.keras.models.load_model("/content/vgg16-KoratCattle.keras")  # แทนที่ด้วยโมเดลของคุณ

def preprocess_image(image):
    """แปลงรูปภาพเป็นขนาดที่โมเดลต้องการ"""
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img = img.resize((224, 224))  # ปรับขนาดให้เข้ากับโมเดลที่ใช้
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติให้กับรูป
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    """รับภาพจาก Wix และส่งค่าทำนายกลับ"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"].read()
    img_array = preprocess_image(file)

    prediction = model.predict(img_array)[0][0]  # ทำนายค่า
    return jsonify({"prediction": float(prediction)})  # ส่งค่าทำนายกลับไป

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

