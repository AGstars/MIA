
# main.py

import numpy as np
import torch
import os
import json
from tqdm import tqdm
from sklearn.base import BaseEstimator
from models.resnet import MNISTResNet18, MNISTResNet9
from utils.data_loader import load_mnist, get_target_loaders, get_shadow_loader
from utils.train_utils import train_model, evaluate_model
from utils.visualization import setup_visdom, update_visdom_plots
from attack.member_inference import train_attack_model
from defense.defense_methods import (
    L2Regularization, DropoutDefense, LabelSmoothing, AdversarialRegularization,
    MixupMMD, ModelStacking, TrustScoreMasking, KnowledgeDistillation
)
from config import ( MODEL_PATH, NUM_SHADOW_MODELS, CHECKPOINTS, L2_LAMBDA, DROPOUT_RATE, LABEL_SMOOTHING, 
                    ADVERSARIAL_EPSILON, MIXUP_ALPHA, MMD_LAMBDA, NUM_STACKED_MODELS, TRUST_SCORE_K, KD_TEMPERATURE, KD_ALPHA)

CHECKPOINT_FILE = 'checkpoint.json'

def save_checkpoint(stage):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'stage': stage}, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)['stage']
    return 0

def ask_continue():
    while True:
        response = input("Do you want to continue to the next stage? (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'.")

def train_target_model(train_loader, test_loader, device, vis):
    model = MNISTResNet18().to(device)
    train_model(model, train_loader, device, vis)
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Target Model Accuracy: {accuracy:.4f}')
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'target_model.pth'))
    return model

def train_shadow_models(train_loader, test_loader, device, vis):
    shadow_models = []
    for i in range(NUM_SHADOW_MODELS):
        print(f'Training Shadow Model {i+1}/{NUM_SHADOW_MODELS}')
        model = MNISTResNet18().to(device)
        train_model(model, train_loader, device, vis)
        shadow_models.append(model)
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'shadow_model_{i}.pth'))
    return shadow_models


    if vis:
        X = list(range(len(defense_names)))
        Y = attack_accuracies

        vis.line(
            Y=[Y],
            X=[X],
            opts=dict(
                xlabel='Defense Method',
                ylabel='Attack Accuracy',
                title='Member Inference Attack Accuracy on Different Defense Methods',
                xtick=True,
                xtickvals=X,
                xticktext=defense_names,
                ytickmin=0,
                ytickmax=1,
                legend=['Attack Accuracy'],
                width=800,
                height=400,
                markers=True,
            ),
            update='append' if len(Y) > 1 else None
        )
    else:
        print("Visdom object is not available. Skipping visualization.")

def perform_attack(model, model_name, shadow_models, target_train_loader, target_test_loader, shadow_train_loader, device):
    print(f"\n{'='*50}")
    print(f"Evaluating attack on {model_name}")
    print(f"{'='*50}")
    
    try:
        attack_model, accuracy, precision, recall, f1, support = train_attack_model(model, shadow_models, target_train_loader, target_test_loader, shadow_train_loader, target_test_loader, device)
        torch.save(attack_model, os.path.join(MODEL_PATH, f'attack_model_{model_name}.pth'))
        print(f"Attack accuracy on {model_name}: {accuracy:.4f}")
        
        macro_avg = [np.mean(precision), np.mean(recall), np.mean(f1)]
        weighted_avg = [np.average(precision, weights=support), 
                        np.average(recall, weights=support), 
                        np.average(f1, weights=support)]
        
        print(f"{'='*50}\n")
        return accuracy, macro_avg, weighted_avg
    except Exception as e:
        print(f"Error during attack on {model_name}: {e}")
        return 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]  # 返回默认值

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vis = setup_visdom()

    # 加载数据
    train_data, test_data = load_mnist()
    target_train_loader, target_test_loader = get_target_loaders(train_data, test_data)
    shadow_train_loader = get_shadow_loader(train_data)

    start_stage = load_checkpoint()

    # 阶段1：训练目标模型
    if start_stage <= 0:
        print("Stage 1: Training target model...")
        target_model = train_target_model(target_train_loader, target_test_loader, device, vis)
        save_checkpoint(1)
        if not ask_continue():
            return
    else:
        print("Loading target model...")
        target_model = MNISTResNet18().to(device)
        target_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model.pth')))

    # 阶段2：训练影子模型
    if start_stage <= 1:
        print("Stage 2: Training shadow models...")
        shadow_models = train_shadow_models(shadow_train_loader, target_test_loader, device, vis)
        save_checkpoint(2)
        if not ask_continue():
            return
    else:
        print("Loading shadow models...")
        shadow_models = []
        for i in range(NUM_SHADOW_MODELS):
            model = MNISTResNet18().to(device)
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'shadow_model_{i}.pth')))
            shadow_models.append(model)

    # 阶段3：进行成员推断攻击
    if start_stage <= 2:
        print("Stage 3: Performing member inference attack...")
        attack_model = train_attack_model(target_model, shadow_models, target_train_loader, target_test_loader, shadow_train_loader, target_test_loader, device)
        torch.save(attack_model, os.path.join(MODEL_PATH, 'attack_model.pth'))
        save_checkpoint(3)
        if not ask_continue():
            return
    else:
        print("Loading attack model...")
        attack_model = torch.load(os.path.join(MODEL_PATH, 'attack_model.pth'))

    # 阶段4：应用防御方法
    if start_stage <= 3:
        print("Stage 4: Applying defense methods...")

    # 阶段4.1：L2正则化    
    if start_stage < CHECKPOINTS['L2_REGULARIZATION']:
        print("Stage 4.1: Applying L2 Regularization...")
        l2_reg = L2Regularization(lambda_reg=L2_LAMBDA)
        target_model_with_l2 = MNISTResNet18().to(device)
        train_model(target_model_with_l2, target_train_loader, device, vis, defense_method=l2_reg)
        accuracy_with_l2 = evaluate_model(target_model_with_l2, target_test_loader, device)
        print(f'Target Model Accuracy with L2 Regularization: {accuracy_with_l2:.4f}')
        torch.save(target_model_with_l2.state_dict(), os.path.join(MODEL_PATH, 'target_model_l2.pth'))
        save_checkpoint(CHECKPOINTS['L2_REGULARIZATION'])
        if not ask_continue():
            return

    # 阶段4.2：Dropout
    if start_stage < CHECKPOINTS['DROPOUT']:
        print("Stage 4.2: Applying Dropout...")
        target_model_with_dropout = DropoutDefense(MNISTResNet18(), dropout_rate=DROPOUT_RATE).to(device)
        train_model(target_model_with_dropout, target_train_loader, device, vis)
        accuracy_with_dropout = evaluate_model(target_model_with_dropout, target_test_loader, device)
        print(f'Target Model Accuracy with Dropout: {accuracy_with_dropout:.4f}')
        torch.save(target_model_with_dropout.state_dict(), os.path.join(MODEL_PATH, 'target_model_dropout.pth'))
        save_checkpoint(CHECKPOINTS['DROPOUT'])
        if not ask_continue():
            return

    # 阶段4.3：标签平滑
    if start_stage < CHECKPOINTS['LABEL_SMOOTHING']:
        print("Stage 4.3: Applying Label Smoothing...")
        label_smoothing = LabelSmoothing(smoothing=LABEL_SMOOTHING)
        target_model_with_ls = MNISTResNet18().to(device)
        train_model(target_model_with_ls, target_train_loader, device, vis, defense_method=label_smoothing)
        accuracy_with_ls = evaluate_model(target_model_with_ls, target_test_loader, device)
        print(f'Target Model Accuracy with Label Smoothing: {accuracy_with_ls:.4f}')
        torch.save(target_model_with_ls.state_dict(), os.path.join(MODEL_PATH, 'target_model_ls.pth'))
        save_checkpoint(CHECKPOINTS['LABEL_SMOOTHING'])
        if not ask_continue():
            return

    # 阶段4.4：对抗正则
    if start_stage < CHECKPOINTS['ADVERSARIAL_REGULARIZATION']:
        print("Stage 4.4: Applying Adversarial Regularization...")
        target_model_with_adv = MNISTResNet18().to(device)
        adv_reg = AdversarialRegularization(target_model_with_adv, epsilon=ADVERSARIAL_EPSILON)
        train_model(target_model_with_adv, target_train_loader, device, vis, defense_method=adv_reg)
        accuracy_with_adv = evaluate_model(target_model_with_adv, target_test_loader, device)
        print(f'Target Model Accuracy with Adversarial Regularization: {accuracy_with_adv:.4f}')
        torch.save(target_model_with_adv.state_dict(), os.path.join(MODEL_PATH, 'target_model_adv.pth'))
        save_checkpoint(CHECKPOINTS['ADVERSARIAL_REGULARIZATION'])
        if not ask_continue():
            return

    # 阶段4.5：Mixup + MMD
    if start_stage < CHECKPOINTS['MIXUP_MMD']:
        print("Stage 4.5: Applying Mixup + MMD...")
        mixup_mmd = MixupMMD(alpha=MIXUP_ALPHA, lambda_mmd=MMD_LAMBDA)
        target_model_with_mixup_mmd = MNISTResNet18().to(device)
        train_model(target_model_with_mixup_mmd, target_train_loader, device, vis, defense_method=mixup_mmd)
        accuracy_with_mixup_mmd = evaluate_model(target_model_with_mixup_mmd, target_test_loader, device)
        print(f'Target Model Accuracy with Mixup + MMD: {accuracy_with_mixup_mmd:.4f}')
        torch.save(target_model_with_mixup_mmd.state_dict(), os.path.join(MODEL_PATH, 'target_model_mixup_mmd.pth'))
        save_checkpoint(CHECKPOINTS['MIXUP_MMD'])
        if not ask_continue():
            return

    # 阶段4.6：模型堆叠
    if start_stage < CHECKPOINTS['MODEL_STACKING']:
        print("Stage 4.6: Applying Model Stacking...")
        stacked_models = [MNISTResNet18().to(device) for _ in range(NUM_STACKED_MODELS)]
        target_model_stacked = ModelStacking(stacked_models).to(device)
        train_model(target_model_stacked, target_train_loader, device, vis)
        accuracy_with_stacking = evaluate_model(target_model_stacked, target_test_loader, device)
        print(f'Target Model Accuracy with Model Stacking: {accuracy_with_stacking:.4f}')
        torch.save(target_model_stacked.state_dict(), os.path.join(MODEL_PATH, 'target_model_stacked.pth'))
        save_checkpoint(CHECKPOINTS['MODEL_STACKING'])
        if not ask_continue():
            return

    # 阶段4.7：信任分数掩蔽
    if start_stage < CHECKPOINTS['TRUST_SCORE_MASKING']:
        print("Stage 4.7: Applying Trust Score Masking...")
        trust_score_masking = TrustScoreMasking(k=TRUST_SCORE_K)
        target_model_with_tsm = MNISTResNet18().to(device)
        train_model(target_model_with_tsm, target_train_loader, device, vis)
        accuracy_with_tsm = evaluate_model(target_model_with_tsm, target_test_loader, device)
        print(f'Target Model Accuracy with Trust Score Masking: {accuracy_with_tsm:.4f}')
        torch.save(target_model_with_tsm.state_dict(), os.path.join(MODEL_PATH, 'target_model_tsm.pth'))
        save_checkpoint(CHECKPOINTS['TRUST_SCORE_MASKING'])
        if not ask_continue():
            return
        
    # 阶段4.8：知识蒸馏
    if start_stage < CHECKPOINTS['KNOWLEDGE_DISTILLATION']:
        print("Stage 4.8: Applying Knowledge Distillation...")

        teacher_model = MNISTResNet18().to(device)
        train_model(teacher_model, target_train_loader, device, vis)

        student_model = MNISTResNet9().to(device)
        kd = KnowledgeDistillation(teacher_model, student_model, temperature=KD_TEMPERATURE, alpha=KD_ALPHA)
        student_model = train_model(student_model, target_train_loader, device, vis, defense_method=kd)
        
        accuracy_with_kd = evaluate_model(student_model, target_test_loader, device)
        print(f'Target Model Accuracy with Knowledge Distillation: {accuracy_with_kd:.4f}')
        torch.save(student_model.state_dict(), os.path.join(MODEL_PATH, 'target_model_kd.pth'))

        teacher_accuracy = evaluate_model(teacher_model, target_test_loader, device)
        print(f'Teacher Model Accuracy: {teacher_accuracy:.4f}')
        
        save_checkpoint(CHECKPOINTS['KNOWLEDGE_DISTILLATION'])
        if not ask_continue():
            return

    print("All defense methods have been applied and evaluated.")

    # 阶段5：对防御后的模型进行成员推断攻击
    if start_stage <= 5:
        print("Stage 5: Performing member inference attack on models with defense...")

    # 加载所有防御模型
    target_model_l2 = MNISTResNet18().to(device)
    target_model_l2.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_l2.pth')))

    target_model_dropout = DropoutDefense(MNISTResNet18(), dropout_rate=DROPOUT_RATE).to(device)
    target_model_dropout.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_dropout.pth')))

    target_model_ls = MNISTResNet18().to(device)
    target_model_ls.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_ls.pth')))

    target_model_adv = MNISTResNet18().to(device)
    target_model_adv.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_adv.pth')))

    target_model_mixup_mmd = MNISTResNet18().to(device)
    target_model_mixup_mmd.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_mixup_mmd.pth')))

    target_model_stacked = ModelStacking([MNISTResNet18().to(device) for _ in range(NUM_STACKED_MODELS)]).to(device)
    target_model_stacked.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_stacked.pth')))

    target_model_tsm = MNISTResNet18().to(device)
    target_model_tsm.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_tsm.pth')))

    target_model_kd = MNISTResNet9().to(device)
    target_model_kd.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'target_model_kd.pth')))

    # 对每个防御模型进行成员推断攻击
    attack_results = {}
    attack_accuracies = []
    macro_avgs = []
    weighted_avgs = []

    defense_models = [
        ("L2 Regularization", target_model_l2),
        ("Dropout", target_model_dropout),
        ("Label Smoothing", target_model_ls),
        ("Adversarial Regularization", target_model_adv),
        ("Mixup + MMD", target_model_mixup_mmd),
        ("Model Stacking", target_model_stacked),
        ("Trust Score Masking", target_model_tsm),
        ("Knowledge Distillation", target_model_kd) 
    ]

    for model_name, model in tqdm(defense_models, desc="Evaluating defense models"):
        attack_acc, macro_avg, weighted_avg = perform_attack(model, model_name, shadow_models, target_train_loader, target_test_loader, shadow_train_loader, device)
        attack_results[model_name] = attack_acc
        attack_accuracies.append(attack_acc)
        macro_avgs.append(macro_avg)
        weighted_avgs.append(weighted_avg)
        
        # 每次攻击后更新 Visdom 图表，不再传入 defense_names
        update_visdom_plots(vis, attack_accuracies, macro_avgs, weighted_avgs)

    print("Final attack results:", attack_results)
    print("Experiment completed.")

if __name__ == "__main__":
    main()