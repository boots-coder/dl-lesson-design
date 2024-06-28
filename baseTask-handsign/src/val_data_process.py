import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# 设定图片和标签路径
image_dir = '../self-data/test'  # 假设图片存储在 images 文件夹中

# 定义目标图像大小
img_size = (64, 64)

# 收集图片和标签
images = []
labels = []

for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        label = int(filename.split('.')[0])  # 假设文件名格式为 'label.png' 或 'label.jpg'
        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        image = image.resize(img_size)  # 调整大小为64x64

        images.append(np.array(image, dtype=np.float32) / 255.0)  # 归一化并转换为 float32
        labels.append(label)

# 将灰度图片绘制出来
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 创建一个包含5个子图的画布
for i in range(5):  # 假设绘制前5张图片
    ax = axes[i]
    ax.imshow(images[i], cmap='gray')  # 使用灰度图像
    ax.set_title(f'Label: {labels[i]}')
    ax.axis('off')  # 不显示坐标轴

plt.show()
# 转换为NumPy数组
X_val = np.array(images, dtype=np.float32)
labels = np.array(labels).reshape(-1, 1)

# 使用OneHotEncoder将标签转换为独热编码格式
encoder = OneHotEncoder(sparse_output=False, categories='auto')
Y_val = encoder.fit_transform(labels).astype(np.float64)

# 检查形状和数据类型
print(f'The data type of the X_val array is: {X_val.dtype}')
print(f'The shape of the X_val array is: {X_val.shape}')
print(f'The data type of the Y_val array is: {Y_val.dtype}')
print(f'The shape of the Y_val array is: {Y_val.shape}')

# 保存为 .npy 文件
np.save('../self-data/test/X_val.npy', X_val)
np.save('../self-data/test/Y_val.npy', Y_val)


print(X_val.dtype)
print(X_val.shape)

print(Y_val.dtype)
print(Y_val.shape)
print(Y_val)