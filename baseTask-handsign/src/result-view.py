import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 定义ResidualBlock和ResNet类（与训练代码中的定义保持一致）

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.fc = nn.Linear(256 * 8 * 8, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 检查MPS是否可用
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

# 加载预训练的模型权重
model.load_state_dict(torch.load('../model/resnet_model-0.9.pth'))
model.to(device)

# 加载验证数据
X_val = np.load('../self-data/test/X_val.npy')  # Shape: (N, 64, 64)
Y_val = np.load('../self-data/test/Y_val.npy')  # Shape: (N, 10)

# 将one-hot编码的标签转换为类别标签
Y_val = np.argmax(Y_val, axis=1)

# 将数据转换为PyTorch张量
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
Y_val = torch.tensor(Y_val, dtype=torch.long)

# 创建验证数据集和加载器
val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 在验证集上评估模型
model.eval()
val_preds = []
val_labels = []
images_batch = []
with torch.no_grad():  # 禁用梯度计算
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())
        images_batch.extend(images.cpu().numpy())

val_accuracy = accuracy_score(val_labels, val_preds)
print(f'Validation accuracy: {val_accuracy}')

# 可视化分类后的图像和标签
def plot_images(images, true_labels, pred_labels, class_names, num_images=10):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = images[i].squeeze()  # 去掉通道维度
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(f'True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}')
    plt.show()

# 类别名称
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

# 绘制前10张图像及其预测和真实标签
plot_images(images_batch, val_labels, val_preds, class_names, num_images=10)
