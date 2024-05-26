import numpy as np
import os
import gzip
import pickle

# 定义路径
MNIST_PATH = r'C:\\OpenSource\\data_set\\MNIST\\MNIST\\raw\\'

# 加载MNIST数据集
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    
    return images, labels

# 加载训练数据
X_train, y_train = load_mnist(MNIST_PATH, kind='train')

# 随机删除数据的函数
def remove_data(X, y, percentage):
    num_samples = int(percentage * len(X))
    indices_to_remove = np.random.choice(len(X), num_samples, replace=False)
    X_new = np.delete(X, indices_to_remove, axis=0)
    y_new = np.delete(y, indices_to_remove, axis=0)
    return X_new, y_new, indices_to_remove

# 删除5%和10%的数据
X_train_, y_train_, removed_indices_ = remove_data(X_train, y_train, 0.50)  # 0.50 = 50%

# 保存修改后的数据集
with open(os.path.join(MNIST_PATH, 'test_50.pkl'), 'wb') as f:
    pickle.dump((X_train_, y_train_), f)

# 打印删除的索引
print("Removed indices for 50% data:", removed_indices_)