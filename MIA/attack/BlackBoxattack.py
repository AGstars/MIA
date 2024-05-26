import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import visdom


BATCH_SIZE = 32                       # 超参数batch大小
EPOCH = 5                             # 总共训练轮数
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 创建GPU运算环境
print(device)

data_transform = {
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

def load_model(path):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 224 * 224, 128)  # 调整输入大小
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

shadow_models = []
num_shadow_models = 5

# 初始化 Visdom
viz = visdom.Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='Training Loss', xlabel='Steps', ylabel='Loss'))

for model_idx in range(num_shadow_models):
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCH):  # 假设训练5个epoch
        for step, (images, labels) in enumerate(shadow_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 Visdom
            viz.line([loss.item()], [epoch * len(shadow_loader) + step], win='train_loss', update='append')

            # 打印输出当前训练的进度
            rate = (step + 1) / len(shadow_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch: {} train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch+1, int(rate * 100), a, b, loss.item()), end="")
        print()  # 换行以便下一次输出

    shadow_models.append(model)

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

    for model in shadow_models:
        shadow_train_predictions.append(get_predictions(model, shadow_loader))
        shadow_test_predictions.append(get_predictions(model, test_loader))

    shadow_train_predictions = np.vstack(shadow_train_predictions)
    shadow_test_predictions = np.vstack(shadow_test_predictions)

    # 创建攻击模型的训练数据
    X_attack_train = np.vstack((shadow_train_predictions, shadow_test_predictions))
    y_attack_train = np.hstack((np.ones(len(shadow_train_predictions)), np.zeros(len(shadow_test_predictions))))

    # 训练攻击模型（使用随机森林分类器）
    attack_model = RandomForestClassifier(n_estimators=100)
    attack_model.fit(X_attack_train, y_attack_train)

    model_paths = [
    "C:/OpenSource/MIA/model/MNIST[0.90]resnet34.pth",
    "C:/OpenSource/MIA/model/MNIST[0.95]resnet34.pth",
    "C:/OpenSource/MIA/model/MNIST[1.0]resnet34.pth"
    ]

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

# 使用攻击模型进行预测
y_attack_pred = attack_model.predict(X_attack_test)

# 计算攻击模型的准确率
attack_acc = accuracy_score(y_attack_test, y_attack_pred)
print(f'Attack model accuracy: {attack_acc:.4f}')