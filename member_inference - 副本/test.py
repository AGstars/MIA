import sys
import os
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from models.resnet import MNISTResNet18
import traceback
import logging
from sklearn.pipeline import Pipeline
import joblib
from utils.data_loader import load_mnist, get_target_loaders
from sklearn.metrics import precision_recall_fscore_support
from utils.train_utils import evaluate_model
from config import MODEL_PATH

class AttackVisualizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("成员推断攻击可视化")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # 模型选择
        model_layout = QHBoxLayout()
        layout.addLayout(model_layout)

        self.attack_model_path = ""
        self.target_model_path = ""

        # 攻击模型选择
        self.attack_model_button = QPushButton("选择攻击模型")
        self.attack_model_button.clicked.connect(self.select_attack_model)
        self.attack_model_label = QLabel("未选择攻击模型")
        model_layout.addWidget(self.attack_model_button)
        model_layout.addWidget(self.attack_model_label)

        # 目标模型选择
        self.target_model_button = QPushButton("选择目标模型")
        self.target_model_button.clicked.connect(self.select_target_model)
        self.target_model_label = QLabel("未选择目标模型")
        model_layout.addWidget(self.target_model_button)
        model_layout.addWidget(self.target_model_label)

        # 开始攻击按钮
        self.start_button = QPushButton("开始攻击")
        self.start_button.clicked.connect(self.start_attack)
        layout.addWidget(self.start_button)

        # 可视化区域
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # 结果显示区域
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

    def select_attack_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择攻击模型", "", "PyTorch Model (*.pth)")
        if file_name:
            self.attack_model_path = file_name
            self.attack_model_label.setText(os.path.basename(file_name))

    def select_target_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择目标模型", "", "PyTorch Model (*.pth)")
        if file_name:
            self.target_model_path = file_name
            self.target_model_label.setText(os.path.basename(file_name))

    def start_attack(self):
        if not self.attack_model_path or not self.target_model_path:
            self.result_label.setText("请先选择攻击模型和目标模型")
            return

        try:
            logging.info("开始加载模型")
            
            # 加载目标模型（PyTorch模型）
            target_model = MNISTResNet18()
            target_model.load_state_dict(torch.load(self.target_model_path, map_location=torch.device('cpu')))
            target_model.eval()

            # 加载攻击模型（同样是PyTorch模型）
            attack_model = MNISTResNet18()
            attack_model.load_state_dict(torch.load(self.attack_model_path, map_location=torch.device('cpu')))
            attack_model.eval()
            
            logging.info("模型加载完成")

            # 加载数据
            logging.info("开始加载数据")
            _, test_loader = load_mnist()
            target_train_loader, target_test_loader = get_target_loaders(test_loader)
            logging.info("数据加载完成")

            # 执行攻击
            logging.info("开始执行攻击")
            accuracy, precision, recall, f1, support = self.simulate_attack(attack_model, target_model, target_train_loader, target_test_loader)

            # 可视化攻击过程
            self.visualize_attack(accuracy)

            # 显示总体评估
            self.show_evaluation(precision, recall, f1, support)

            logging.info("攻击执行完成")

        except Exception as e:
            error_msg = f"发生错误: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.result_label.setText(error_msg)
            print(error_msg)  # 也在控制台打印错误信息

    def simulate_attack(self, attack_model, target_model, target_train_loader, target_test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attack_model.to(device)
        target_model.to(device)

        criterion = torch.nn.BCELoss()
        all_labels = []
        all_predictions = []

        # 模拟攻击过程
        for i, ((train_data, _), (test_data, _)) in enumerate(zip(target_train_loader, target_test_loader)):
            # 训练数据
            train_data = train_data.to(device)
            train_outputs = target_model(train_data)
            train_attack_inputs = torch.cat((train_outputs, train_data), dim=1)
            train_attack_outputs = attack_model(train_attack_inputs)
            all_labels.extend([1] * train_data.size(0))
            all_predictions.extend(train_attack_outputs.cpu().detach().numpy())

            # 测试数据
            test_data = test_data.to(device)
            test_outputs = target_model(test_data)
            test_attack_inputs = torch.cat((test_outputs, test_data), dim=1)
            test_attack_outputs = attack_model(test_attack_inputs)
            all_labels.extend([0] * test_data.size(0))
            all_predictions.extend(test_attack_outputs.cpu().detach().numpy())

            # 更新可视化（每10个批次更新一次）
            if i % 10 == 0:
                current_accuracy = np.mean((np.array(all_predictions) > 0.5) == np.array(all_labels))
                self.visualize_attack(current_accuracy)
                QApplication.processEvents()  # 允许 GUI 更新

        # 计算最终指标
        all_labels = np.array(all_labels)
        all_predictions = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = np.mean(all_predictions == all_labels)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions)

        return accuracy, precision, recall, f1, support

    def visualize_attack(self, accuracy):
        self.ax.clear()
        self.ax.plot([0, 1], [0, accuracy], 'r-')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('攻击进度')
        self.ax.set_ylabel('准确率')
        self.ax.set_title('攻击过程可视化')
        self.canvas.draw()

    def show_evaluation(self, precision, recall, f1, support):
        eval_text = f"总体评估:\n"
        eval_text += f"宏平均 (Macro Avg)    精确度: {np.mean(precision):.2f}  召回率: {np.mean(recall):.2f}  F1-score: {np.mean(f1):.2f}\n"
        eval_text += f"加权平均 (Weighted Avg) 精确度: {np.average(precision, weights=support):.2f}  "
        eval_text += f"召回率: {np.average(recall, weights=support):.2f}  "
        eval_text += f"F1-score: {np.average(f1, weights=support):.2f}"
        self.result_label.setText(eval_text)

def main():
    app = QApplication(sys.argv)
    window = AttackVisualizationWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()