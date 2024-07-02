import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_mean_ap(y_true, y_scores, num_classes):
    """
    计算多分类任务下的平均精度 (AP)

    参数:
    y_true (np.array): 真实标签，形状为 (num_samples, num_classes)
    y_scores (np.array): 预测得分，形状为 (num_samples, num_classes)
    num_classes (int): 类别数

    返回:
    mean_ap (float): 平均精度
    ap_per_class (list): 每个类别的AP值
    """
    ap_per_class = []

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        ap_per_class.append(ap)

    mean_ap = np.mean(ap_per_class)

    return mean_ap, ap_per_class


# 示例数据
num_samples = 100
num_classes = 5
y_true = np.random.randint(2, size=(num_samples, num_classes))
y_scores = np.random.rand(num_samples, num_classes)

# 计算平均精度
mean_ap, ap_per_class = calculate_mean_ap(y_true, y_scores, num_classes)

print(f"Mean AP: {mean_ap}")
print(f"AP per class: {ap_per_class}")


import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 示例数据
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

# 初始化 OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)

# 转换为独热编码
one_hot_encoded = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

print(one_hot_encoded)