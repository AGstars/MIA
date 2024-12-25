
# attack/member_inference.py

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

def generate_attack_data(model, data_loader, is_member, device):
    model.eval()
    attack_data = []
    attack_labels = []
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc=f"Generating {'member' if is_member else 'non-member'} data"):
            data = data.to(device)
            outputs = model(data)
            features = torch.cat((outputs, data.view(data.size(0), -1)), dim=1).cpu().numpy()
            attack_data.extend(features)
            attack_labels.extend([is_member] * data.size(0))
    return np.array(attack_data), np.array(attack_labels)

def train_attack_model(target_model, shadow_models, target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader, device):
    print("Generating attack data...")
    
    # 生成目标模型的成员和非成员数据
    member_data, member_labels = generate_attack_data(target_model, target_train_loader, is_member=1, device=device)
    non_member_data, non_member_labels = generate_attack_data(target_model, target_test_loader, is_member=0, device=device)
    
    # 生成影子模型的成员和非成员数据
    shadow_member_data = []
    shadow_non_member_data = []
    for i, shadow_model in enumerate(shadow_models):
        print(f"Processing shadow model {i+1}/{len(shadow_models)}")
        shadow_member, _ = generate_attack_data(shadow_model, shadow_train_loader, is_member=1, device=device)
        shadow_non_member, _ = generate_attack_data(shadow_model, shadow_test_loader, is_member=0, device=device)
        
        shadow_member_data.append(shadow_member)
        shadow_non_member_data.append(shadow_non_member)
    
    # 组合所有数据
    attack_data = np.vstack([
        member_data, 
        non_member_data, 
        *shadow_member_data, 
        *shadow_non_member_data
    ])
    
    attack_labels = np.hstack([
        member_labels,
        non_member_labels,
        np.ones(sum(len(data) for data in shadow_member_data)),
        np.zeros(sum(len(data) for data in shadow_non_member_data))
    ])
    
    # 训练攻击模型
    print("Training attack model...")
    X_train, X_test, y_train, y_test = train_test_split(
        attack_data, 
        attack_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=attack_labels
    )
    
    # 创建包含数据预处理和模型训练的管道
    attack_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # 训练模型
    attack_model.fit(X_train, y_train)
    
    # 评估攻击模型
    y_pred = attack_model.predict(X_test)
    
    # 计算详细的评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    # 打印详细的评估结果
    print("详细评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.2f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n每个类别的详细指标:")
    for i, class_label in enumerate(["非成员", "成员"]):
        print(f"类别 {i} ({class_label}):")
        print(f"  精确度 (Precision): {precision[i]:.2f}")
        print(f"  召回率 (Recall): {recall[i]:.2f}")
        print(f"  F1-score: {f1[i]:.2f}")
        print(f"  Support: {support[i]}")
    
    print("\n总体评估:")
    print(f"宏平均 (Macro Avg)    精确度: {np.mean(precision):.2f}  召回率: {np.mean(recall):.2f}  F1-score: {np.mean(f1):.2f}")
    print(f"加权平均 (Weighted Avg) 精确度: {np.average(precision, weights=support):.2f}  "
          f"召回率: {np.average(recall, weights=support):.2f}  "
          f"F1-score: {np.average(f1, weights=support):.2f}")
    
    return attack_model, accuracy, precision, recall, f1, support