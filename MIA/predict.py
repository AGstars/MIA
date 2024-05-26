import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from model import resnet34 


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),  # 调整图像尺寸
                                    transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为三通道
                                    transforms.ToTensor(),  # 转换为张量
                                    transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)])

    # load image
    image_root = os.path.abspath(os.path.join(os.getcwd(), "C:/OpenSource/data_test/"))  # get data root path
    img_path = os.path.join(image_root, "num9.jpg")  # flower data set path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'C:/OpenSource/MIA/MINSTclass_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=1000).to(device)

    # load model weights
    weights_path = "C:/OpenSource/MIA/model/MNIST[0.95]resnet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    plt.title(print_res)
    plt.show()
    for i in range(len(predict)):
        class_name = class_indict[str(i)]
        prob = predict[i].numpy()
        print(f"class: {class_name:10}   prob: {prob:.3}")

if __name__ == '__main__':
    main()