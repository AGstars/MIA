import torch
from model import resnet34
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Compose, Resize, Grayscale
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# 数据转化为tensor格式
data_transform = Compose([
    Resize((224, 224)),  # 调整图像尺寸以适应ResNet18的输入
    Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
    ToTensor(),
    Normalize((0.1307,) * 3, (0.3081,) * 3)  # 使用MNIST的均值和标准差
])

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，将模型数据转到GPU
model = resnet34().to(device)
model.load_state_dict(torch.load("C:/OpenSource/MIA/model/MNIST[0.95]resnet34.pth", map_location=torch.device('cpu')))

# 获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 加载并处理图片
image_root = os.path.abspath(os.path.join(os.getcwd(), "C:/OpenSource/data_test/"))  # get data root path
image_path = os.path.join(image_root, "num9.jpg")  # flower data set path

img = Image.open(image_path)  # 打开图片
img_tensor = data_transform(img).unsqueeze(0).to(device)  # 预处理并添加批次维度


# 执行推理
model.eval()
with torch.no_grad():
    pred = model(img_tensor)
    predicted = classes[torch.argmax(pred[0])]
    accuracy = torch.nn.functional.softmax(pred, dim=1)[0][torch.argmax(pred[0])].item()

# 显示预测结果
fig, ax = plt.subplots()
img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C) 格式
img = (img * 0.3081 + 0.1307) * 255  # 反归一化
img = np.clip(img, 0, 255).astype(np.uint8)
ax.imshow(img, cmap="gray")
ax.set_title(f"Predicted: {predicted}   Prob: {accuracy:.2f}")
ax.axis('off')
plt.show()