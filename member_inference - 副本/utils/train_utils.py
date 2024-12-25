
# /utils/train_utils.py

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.base import BaseEstimator
from config import EPOCHS, LEARNING_RATE
from defense.defense_methods import (
    L2Regularization, DropoutDefense, LabelSmoothing, AdversarialRegularization,
    MixupMMD, ModelStacking, TrustScoreMasking, KnowledgeDistillation
)

def train_model(model, train_loader, device, vis=None, defense_method=None):

    if isinstance(defense_method, KnowledgeDistillation):
        return train_with_knowledge_distillation(defense_method, train_loader, device, vis)

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass and loss calculation
            if isinstance(defense_method, (MixupMMD, AdversarialRegularization)):
                loss = defense_method(data, target, model)
                output = model(data)
            elif isinstance(defense_method, LabelSmoothing):
                output = model(data)
                loss = defense_method(output, target)
            elif isinstance(defense_method, (ModelStacking)):
                output, loss = defense_method(data, target, model)
            else:
                output = model(data)
                loss = criterion(output, target)

            # Additional regularization or defense steps
            if isinstance(defense_method, L2Regularization):
                loss += defense_method(model)
            elif isinstance(defense_method, DropoutDefense):
                defense_method.apply(model)
            elif isinstance(defense_method, TrustScoreMasking):
                loss = defense_method.apply_masking(loss, output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

        if vis:
            vis.line(X=[epoch], Y=[running_loss / len(train_loader)], win='loss', update='append' if epoch > 0 else None, opts=dict(title='Training Loss'))
            vis.line(X=[epoch], Y=[100. * correct / total], win='accuracy', update='append' if epoch > 0 else None, opts=dict(title='Training Accuracy'))

    return model

def train_with_knowledge_distillation(kd, train_loader, device, vis=None):
    student_model = kd.student_model
    student_model.to(device)
    student_model.train()
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            loss = kd(data, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            output = student_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

        if vis:
            vis.line(X=[epoch], Y=[running_loss / len(train_loader)], win='loss', update='append' if epoch > 0 else None, opts=dict(title='Training Loss'))
            vis.line(X=[epoch], Y=[100. * correct / total], win='accuracy', update='append' if epoch > 0 else None, opts=dict(title='Training Accuracy'))

    return student_model

""" def evaluate_model(*args, **kwargs):
    if len(args) == 3 and isinstance(args[1], torch.utils.data.DataLoader):
        return evaluate_standard_model(*args, **kwargs)
    elif len(args) == 6 and isinstance(args[1], torch.nn.Module):
        return evaluate_attack_model(*args, **kwargs)
    else:
        raise ValueError("Invalid arguments for evaluate_model") """

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

""" def evaluate_attack_model(attack_model, target_model, train_loader, test_loader, device, n_features):
    target_model.to(device)
    target_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Evaluate on training data (member)
        for data, _ in tqdm(train_loader, desc='Evaluating on train data'):
            data = data.to(device)
            target_outputs = target_model(data)
            
            # Flatten the data and outputs
            data_flattened = data.view(data.size(0), -1)
            target_outputs_flattened = target_outputs.view(target_outputs.size(0), -1)
            
            attack_inputs = torch.cat((target_outputs_flattened, data_flattened), dim=1).cpu().numpy()
            
            # Ensure the number of features matches what the attack model expects
            if attack_inputs.shape[1] > n_features:
                attack_inputs = attack_inputs[:, :n_features]
            elif attack_inputs.shape[1] < n_features:
                attack_inputs = np.pad(attack_inputs, ((0, 0), (0, n_features - attack_inputs.shape[1])))
            
            predicted = attack_model.predict(attack_inputs)
            total += data.size(0)
            correct += (predicted == 1).sum()  # 1 表示成员

        # Evaluate on test data (non-member)
        for data, _ in tqdm(test_loader, desc='Evaluating on test data'):
            data = data.to(device)
            target_outputs = target_model(data)
            
            # Flatten the data and outputs
            data_flattened = data.view(data.size(0), -1)
            target_outputs_flattened = target_outputs.view(target_outputs.size(0), -1)
            
            attack_inputs = torch.cat((target_outputs_flattened, data_flattened), dim=1).cpu().numpy()
            
            # Ensure the number of features matches what the attack model expects
            if attack_inputs.shape[1] > n_features:
                attack_inputs = attack_inputs[:, :n_features]
            elif attack_inputs.shape[1] < n_features:
                attack_inputs = np.pad(attack_inputs, ((0, 0), (0, n_features - attack_inputs.shape[1])))
            
            predicted = attack_model.predict(attack_inputs)
            total += data.size(0)
            correct += (predicted == 0).sum()  # 0 表示非成员

    accuracy = correct / total
    return accuracy """