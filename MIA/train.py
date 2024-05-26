import visdom
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from matplotlib import pyplot as plt
from model import resnet34
from torch.nn import CrossEntropyLoss
from torch import optim
import torchvision
import torch.utils.data as Data
from PIL import Image

# 自定义 CIFAR-10 数据集类
class CustomCIFAR10(Dataset):
    def __init__(self, data, targets, transform=None):
        super(CustomCIFAR10, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # 将图像转换为PIL格式
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # 确保标签是 Long 类型
        return img, torch.tensor(target, dtype=torch.long)
    
def main():
    BATCH_SIZE = 32                       # 超参数batch大小
    EPOCH = 2                             # 总共训练轮数
    save_path = "./CIFAR10[0.50]resnet34.pth"      # 模型权重参数保存位置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 创建GPU运算环境
    print(device)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    try:
        with open('C:/OpenSource/data_set/CIFAR10/cifar-10-batches-py/train_50.pkl', 'rb') as file:
            data, targets = pickle.load(file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 检查数据和标签的类型，确保它们是 numpy 数组或 torch 张量，如果需要，进行转换
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.uint8)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets, dtype=np.int64)

    # 加载训练和测试数据集
    train_dataset = CustomCIFAR10(data=data, targets=targets, transform=data_transform["train"])
    test_dataset = torchvision.datasets.CIFAR10(root='C:/OpenSource/data_set/CIFAR10/', train=False, download=True, transform=data_transform["val"])

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # 创建一个visdom，将训练测试情况可视化
    #viz = visdom.Visdom()

    # 测试函数，传入模型和数据读取接口
    def evalute(model, loader):
        # correct为总正确数量，total为总测试数量
        correct = 0
        total = len(loader.dataset)
        # 取测试数据
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # validation和test过程不需要反向传播
            model.eval()
            with torch.no_grad():
                out = model(x)       # 计算测试数据的输出logits
                # 计算出out在第一维度上最大值对应编号，得模型的预测值
                prediction = out.argmax(dim=1)
            # 预测正确的数量correct
            correct += torch.eq(prediction, y).float().sum().item()
        # 最终返回正确率
        return correct / total


    net = resnet34()
    net.to(device)                                      # 实例化网络模型并送入GPU
    # net.load_state_dict(torch.load(save_path))          # 使用上次训练权重接着训练
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001) # 定义优化器
    loss_function = CrossEntropyLoss()                  # 多分类问题使用交叉熵损失函数

    best_acc, best_epoch = 0.0, 0                       # 最好准确度，出现的轮数
    global_step = 0                                     # 全局的step步数，用于画图
    for epoch in range(EPOCH):      
        running_loss = 0.0                              # 一次epoch的总损失
        net.train()                                     # 开始训练
        for step, (images, labels) in enumerate(train_dataloader, start=0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()                 # 将一个epoch的损失累加
            # 打印输出当前训练的进度
            rate = (step + 1) / len(train_dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch: {} train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch+1, int(rate * 100), a, b, loss), end="")
            # 记录test的loss
            #viz.line([loss.item()], [global_step], win='loss', update='append')
            # 每次记录之后将横轴x的值加一
            global_step += 1

        # 在每一个epoch

        if epoch % 1 == 0:
            test_acc = evalute(net, test_dataloader)
            print("  epoch{} test acc:{} best epoch:{}".format(epoch+1, test_acc,best_epoch))
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(net.state_dict(), save_path)
                #viz.line([test_acc], [global_step], win='test_acc', update='append')
    print("Finish")

if __name__ == '__main__':
    main()