
# config.py

import os

# 数据集相关配置
TRAIN_SIZE = 50000
TEST_SIZE = 10000
SHADOW_SIZE = 50000

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mnist')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved')

# 确保必要的目录存在
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 模型训练相关配置
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_SHADOW_MODELS = 5

# 防御方法参数
L2_LAMBDA = 0.01
DROPOUT_RATE = 0.5
LABEL_SMOOTHING = 0.1
ADVERSARIAL_EPSILON = 0.1
MIXUP_ALPHA = 1.0
MMD_LAMBDA = 0.1
NUM_STACKED_MODELS = 3
TRUST_SCORE_K = 3
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.5

# 检查点
CHECKPOINTS = {
    'TARGET_MODEL': 4.0,
    'L2_REGULARIZATION': 4.1,
    'DROPOUT': 4.2,
    'LABEL_SMOOTHING': 4.3,
    'ADVERSARIAL_REGULARIZATION': 4.4,
    'MIXUP_MMD': 4.5,
    'MODEL_STACKING': 4.6,
    'TRUST_SCORE_MASKING': 4.7,
    'KNOWLEDGE_DISTILLATION': 4.8
}

# Visdom配置
VISDOM_SERVER = 'http://localhost'
VISDOM_PORT = 8097