import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score

# 定义路径
train_dir = '../NEU Metal Surface Defects Data/train'
val_dir = '../NEU Metal Surface Defects Data/valid'
test_dir = '../NEU Metal Surface Defects Data/test'

# 检查并打印路径下的目录和文件
def list_directory_contents(path, description):
    try:
        contents = os.listdir(path)
        print(f"{description}: {contents}")
    except FileNotFoundError:
        print(f"Error: {path} not found.")
    except NotADirectoryError:
        print(f"Error: {path} is not a directory.")
    except PermissionError:
        print(f"Error: Permission denied for {path}.")

base_dir = '../NEU Metal Surface Defects Data'
list_directory_contents(base_dir, "Path")
list_directory_contents(train_dir, "Train")
list_directory_contents(test_dir, "Test")
list_directory_contents(val_dir, "Validation")

print("Inclusion Defect")
print("Training Images:", len(os.listdir(train_dir+'/'+'Inclusion')))
print("Testing Images:", len(os.listdir(test_dir+'/'+'Inclusion')))
print("Validation Images:", len(os.listdir(val_dir+'/'+'Inclusion')))

# Rescaling all Images by 1./255

# 创建训练和验证的 ImageDataGenerator 实例
train_datagen = ImageDataGenerator(
    rescale=1./255,             # 将像素值从 0-255 缩放到 0-1
    shear_range=0.2,            # 随机应用剪切变换
    zoom_range=0.2,             # 随机缩放图像
    horizontal_flip=True        # 随机水平翻转图像
)



test_datagen = ImageDataGenerator(rescale=1./255)

# Training images are put in batches of 10
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

# Validation images are put in batches of 10
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'recall'])

callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 32,
        epochs=2,
        validation_data=validation_generator,
        callbacks=[callbacks],
        verbose=1, shuffle=True)

# 评估模型在验证集上的表现
val_loss, val_accuracy, val_recall = model.evaluate(validation_generator)

print(f"验证集的准确率: {val_accuracy:.4f}")
print(f"验证集的召回率: {val_recall:.4f}")

# 计算 ap 和 map
def calculate_mean_ap(y_true, y_scores, num_classes):
    ap_per_class = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        ap_per_class.append(ap)
    mean_ap = np.mean(ap_per_class)
    return mean_ap, ap_per_class

num_classes = 6
# num_samples = len(validation_generator.labels)
# print(num_samples)
# print(validation_generator.labels)
#
#
# print("validation-generator{}", validation_generator)
# print(model.predict(validation_generator).shape)
#
# # 初始化 OneHotEncoder
# one_hot_encoder = OneHotEncoder(sparse_output=False)
# one_hot_encoded_labels = one_hot_encoder.fit_transform(validation_generator.labels.reshape(-1, 1))
# y_true = one_hot_encoded_labels
#
# y_scores = model.predict(validation_generator)
# print(y_scores)


# mean_ap, ap_per_class = calculate_mean_ap(y_true, y_scores, num_classes)
#
# print(f"Mean AP: {mean_ap}")
# print(f"AP per class: {ap_per_class}")
# Loading file names & their respective target labels into numpy array
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
x_test, y_test,target_labels = load_dataset(test_dir)
no_of_classes = len(np.unique(y_test))

# y_test = np_utils.to_categorical(y_test,no_of_classes)
from keras.utils import to_categorical
# print(y_test)
y_test = to_categorical(y_test, num_classes=no_of_classes)
#
# print(y_test.shape)
# print(y_test)
from keras.preprocessing.image import array_to_img, img_to_array, load_img
def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

x_test = x_test.astype('float32')/255
# Plotting Random Sample of test images, their predicted labels, and ground truth
y_pred = model.predict(x_test)
print(y_pred)


mean_ap, ap_per_class = calculate_mean_ap(y_test, y_pred, num_classes)

print(f"Mean AP: {mean_ap}")
print(f"AP per class: {ap_per_class}")

classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Plotting- ap 和 map 的可视化展示

plt.figure(figsize=(10, 6))
bars = plt.bar(classes, ap_per_class, color='skyblue')
plt.axhline(y=mean_ap, color='r', linestyle='--', label=f'Mean AP: {mean_ap:.2f}')
plt.xlabel('Classes')
plt.ylabel('AP')
plt.title('AP per Class and Mean AP')
plt.legend()

# Adding text labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.show()