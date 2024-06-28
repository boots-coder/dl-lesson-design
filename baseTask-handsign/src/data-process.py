import numpy as np
import matplotlib.pyplot as plt

# 读取.npy文件
data = np.load('../data/X.npy')
labels = np.load('../data/Y.npy')

# 获取数据类型和形状
data_type = data.dtype
data_shape = data.shape
labels_type = labels.dtype
labels_shape = labels.shape

print(f"The data type of the X.npy file is: {data_type}")
print(f"The shape of the X.npy file is: {data_shape}")
print(f"The data type of the Y.npy file is: {labels_type}")
print(f"The shape of the Y.npy file is: {labels_shape}")
print(labels)
# 用于记录已遇到的标签
encountered_labels = set()

# 打印出每个不同标签对应的图片和独热值
for i in range(len(labels)):
    # 获取当前标签的索引
    label_index = np.argmax(labels[i])

    if label_index not in encountered_labels:
        # 记录已遇到的标签
        encountered_labels.add(label_index)

        # 打印标签及其对应的图片和独热值
        print(f"Label index: {label_index}, One-hot value: {labels[i]}")
        plt.imshow(data[i], cmap='gray')
        plt.title(f"Label index: {label_index}")
        plt.show()

    # 如果已经找到所有10种不同的标签，则退出循环
    if len(encountered_labels) >= 10:
        break
        
        
