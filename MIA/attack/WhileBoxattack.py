import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

NUM_CLASSES = 1000
BATCH_SIZE = 32                       # 超参数batch大小
EPOCH = 5                             # 总共训练轮数
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据变换
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

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=True, download=True, transform=data_transform["train"])
test_dataset = datasets.MNIST(root='C:/OpenSource/data_set/MNIST', train=False, download=True, transform=data_transform["val"])

# 拆分数据集
train_size = int(0.5 * len(train_dataset))
shadow_size = len(train_dataset) - train_size
train_dataset, shadow_dataset = torch.utils.data.random_split(train_dataset, [train_size, shadow_size])

# 数据加载器
shadow_loader = torch.utils.data.DataLoader(shadow_dataset, BATCH_SIZE, shuffle=True)
attack_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

# 定义和加载预训练的 ResNet34 模型
def load_model(path):
    model = models.resnet34(weights=None)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    model.eval()
    return model

model_paths = [
    "C:/OpenSource/MIA/model/MNIST[1.0]resnet34.pth"
]

shadow_models = [load_model(path) for path in model_paths]

# 提取中间结果
def get_intermediate_activations(model, loader, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    batch_idx = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = []

            def hook(module, input, output):
                outputs.append(output.cpu().numpy().reshape(output.size(0), -1))

            hooks = []
            for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                hooks.append(layer.register_forward_hook(hook))

            _ = model(images)

            for hook in hooks:
                hook.remove()

            outputs = np.concatenate(outputs, axis=1)
            np.save(os.path.join(output_dir, f"{prefix}_batch_{batch_idx}.npy"), outputs)
            batch_idx += 1

# 获取影子模型的中间激活值并保存到磁盘
output_dir = './activations'
for i, model in enumerate(shadow_models):
    get_intermediate_activations(model, shadow_loader, output_dir, f'shadow_train_{i}')
    get_intermediate_activations(model, test_loader, output_dir, f'shadow_test_{i}')

# 加载保存的激活值
def load_saved_activations(output_dir, prefix, num_batches):
    activations = []
    for batch_idx in range(num_batches):
        batch_activations = np.load(os.path.join(output_dir, f"{prefix}_batch_{batch_idx}.npy"))
        activations.append(batch_activations)
    return np.concatenate(activations, axis=0)

# 估计影子模型的批次数量
num_batches_shadow_train = len(shadow_loader.dataset) // BATCH_SIZE
num_batches_shadow_test = len(test_loader.dataset) // BATCH_SIZE

# 加载影子模型的激活值
shadow_train_activations = []
shadow_test_activations = []

for i in range(len(shadow_models)):
    shadow_train_activations.append(load_saved_activations(output_dir, f'shadow_train_{i}', num_batches_shadow_train))
    shadow_test_activations.append(load_saved_activations(output_dir, f'shadow_test_{i}', num_batches_shadow_test))

# 把所有影子模型的激活值合并起来
shadow_train_activations = np.vstack(shadow_train_activations)
shadow_test_activations = np.vstack(shadow_test_activations)

# 构建和训练攻击模型
# 创建攻击模型的训练数据
X_attack_train = np.vstack((shadow_train_activations, shadow_test_activations))
y_attack_train = np.hstack((np.ones(len(shadow_train_activations)), np.zeros(len(shadow_test_activations))))

# 训练攻击模型
attack_model = RandomForestClassifier(n_estimators=100)
attack_model.fit(X_attack_train, y_attack_train)

# 准备攻击模型的测试数据
shadow_attack_activations = []
shadow_test_activations = []

for model in shadow_models:
    shadow_attack_activations.append(get_intermediate_activations(model, attack_loader))
    shadow_test_activations.append(get_intermediate_activations(model, test_loader))

# 把所有影子模型的激活值合并起来
shadow_attack_activations = np.vstack(shadow_attack_activations)
shadow_test_activations = np.vstack(shadow_test_activations)

# 准备攻击模型的测试数据
X_attack_test = np.vstack((shadow_attack_activations, shadow_test_activations))
y_attack_test = np.hstack((np.ones(len(shadow_attack_activations)), np.zeros(len(shadow_test_activations))))

# 评估攻击模型
y_attack_pred = attack_model.predict(X_attack_test)
accuracy = accuracy_score(y_attack_test, y_attack_pred)
print(f'[White Box]Attack model accuracy  using with [MNIST[0.95]resnet34.pth]: {accuracy:.4f}')