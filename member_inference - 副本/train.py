
# train.py

import torch
from models.resnet import MNISTResNet18
from utils.data_loader import load_mnist, get_target_loaders
from utils.train_utils import train_model, evaluate_model
from utils.visualization import setup_visdom
from config import MODEL_PATH
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vis = setup_visdom()

    train_data, test_data = load_mnist()
    train_loader, test_loader = get_target_loaders(train_data, test_data)

    model = MNISTResNet18().to(device)
    
    train_model(model, train_loader, device, vis)
    
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Model Accuracy: {accuracy:.4f}')

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'target_model.pth'))

if __name__ == "__main__":
    main()