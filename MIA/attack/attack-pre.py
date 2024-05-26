import numpy as np
import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from net import MyLeNet5
sys.path.pop()

NUM_CLASSES = 1000
BATCH_SIZE = 32                       # 超参数batch大小
EPOCH = 5                             # 总共训练轮数
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 创建GPU运算环境
print(device)

# ResNet34 MNIST
""" data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),  # 调整图像尺寸
                                    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
                                    transforms.ToTensor(),  # 转换为张量
                                    transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # 调整图像尺寸
                                    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
                                    transforms.ToTensor(),  # 转换为张量
                                    transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)])}
# 加载测试数据集
train_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=True, download=True, transform=data_transform["train"])
test_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=False, download=True, transform=data_transform["val"])

# 拆分数据集
train_size = int(0.5 * len(train_dataset))
shadow_size = len(train_dataset) - train_size
train_dataset, shadow_dataset = torch.utils.data.random_split(train_dataset, [train_size, shadow_size])

shadow_loader = torch.utils.data.DataLoader(shadow_dataset, BATCH_SIZE, shuffle=True)
attack_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
 """

# Resnet34 CIFAR10
""" data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([ transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])} """

# LeNet5 CIFAR10
""" data_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((28, 28)),  # 将图像调整为28x28
                                        transforms.Normalize((0.5), (0.5))]) """

# 加载训练和测试数据集（CIFAR10）
""" train_dataset = datasets.CIFAR10(root='C:/OpenSource/data_set/CIFAR10/', train=True, download=True, transform=data_transform)
test_dataset = datasets.CIFAR10(root='C:/OpenSource/data_set/CIFAR10/', train=False, download=True, transform=data_transform)

# 拆分数据集
train_size = int(0.5 * len(train_dataset))
shadow_size = len(train_dataset) - train_size
train_dataset, shadow_dataset = torch.utils.data.random_split(train_dataset, [train_size, shadow_size])

shadow_loader = torch.utils.data.DataLoader(dataset=shadow_dataset, shuffle=True, batch_size=BATCH_SIZE)
attack_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE) """

# 加载测试数据集（MNIST）
train_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=False, download=True, transform=transforms.ToTensor())

# 拆分数据集
train_size = int(0.5 * len(train_dataset))
shadow_size = len(train_dataset) - train_size
train_dataset, shadow_dataset = torch.utils.data.random_split(train_dataset, [train_size, shadow_size])

shadow_loader = torch.utils.data.DataLoader(shadow_dataset, BATCH_SIZE, shuffle=True)
attack_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# 定义和加载预训练的 ResNet34 模型
""" def load_model(path):
    model = models.resnet34(weights=None)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES) 
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()  # 切换到评估模式
    return model """

# 定义和加载预训练的 LeNet5 模型
def load_model(path):
    model = MyLeNet5()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # 切换到评估模式
    return model

model_paths = [
    "C:/OpenSource/MIA/model/lenet/MNIST[0.50]LeNet5.pth"
]

shadow_models = [load_model(path) for path in model_paths]

def get_predictions(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)

shadow_train_predictions = []
shadow_test_predictions = []

print('Finish get_predictions')
for model in shadow_models:
    shadow_train_predictions.append(get_predictions(model, shadow_loader))
    shadow_test_predictions.append(get_predictions(model, test_loader))

shadow_train_predictions = np.vstack(shadow_train_predictions)
shadow_test_predictions = np.vstack(shadow_test_predictions)

# 创建攻击模型的训练数据
X_attack_train = np.vstack((shadow_train_predictions, shadow_test_predictions))
y_attack_train = np.hstack((np.ones(len(shadow_train_predictions)), np.zeros(len(shadow_test_predictions))))
print('Finish create attack model train data')

# 训练攻击模型（使用随机森林分类器）
attack_model = RandomForestClassifier(n_estimators=100)
attack_model.fit(X_attack_train, y_attack_train)
print('Finish attack model trainging')


# 评估攻击模型
def evaluate_attack(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)

train_preds, train_labels = evaluate_attack(model, attack_loader)
test_preds, test_labels = evaluate_attack(model, test_loader)

X_attack_test = np.vstack((train_preds, test_preds))
y_attack_test = np.hstack((np.ones(len(train_preds)), np.zeros(len(test_preds))))
print('Finish attack model predict')

# 使用攻击模型进行预测
y_attack_pred = attack_model.predict(X_attack_test)

# 计算攻击模型的准确率
attack_acc = accuracy_score(y_attack_test, y_attack_pred)
print(f'Attack model accuracy  using with [MNIST[0.50]LeNet5.pth]: {attack_acc:.4f}')