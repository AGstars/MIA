
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Regularization:
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg

    def __call__(self, model):
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.lambda_reg * l2_loss

class DropoutDefense(nn.Module):
    def __init__(self, model, dropout_rate=0.5):
        super(DropoutDefense, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.model(x)
        return self.dropout(x)

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        target = target.unsqueeze(1)
        one_hot = torch.zeros_like(pred).scatter(1, target, 1)
        smoothed_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        return F.kl_div(log_prob, smoothed_target, reduction='batchmean')

class AdversarialRegularization(nn.Module):
    def __init__(self, model, epsilon=0.1):
        super(AdversarialRegularization, self).__init__()
        self.model = model
        self.epsilon = epsilon

    def forward(self, x):
        x.requires_grad = True
        output = self.model(x)
        return output

    def calculate_loss(self, x, y):
        output = self(x)
        loss = F.cross_entropy(output, y)
        loss.backward(retain_graph=True)

        adv_x = x + self.epsilon * x.grad.sign()
        adv_output = self.model(adv_x)
        adv_loss = F.kl_div(adv_output.log_softmax(dim=1), output.softmax(dim=1), reduction='batchmean')
        return loss + adv_loss

class MixupMMD(nn.Module):
    def __init__(self, alpha=1.0, lambda_mmd=0.1):
        super(MixupMMD, self).__init__()
        self.alpha = alpha
        self.lambda_mmd = lambda_mmd

    def forward(self, x, y, model):
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Mixup
        lambda_ = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().to(x.device)
        mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
        
        y_a, y_b = y, y[index]
        mixed_y = lambda_ * F.one_hot(y_a, num_classes=10).float() + (1 - lambda_) * F.one_hot(y_b, num_classes=10).float()
        
        # Forward pass
        output = model(mixed_x)
        
        # Compute Mixup loss
        loss = -torch.sum(mixed_y * F.log_softmax(output, dim=1), dim=1).mean()
        
        # Compute MMD loss
        feature_original = model.get_features(x)
        feature_mixed = model.get_features(mixed_x)
        mmd_loss = self.mmd(feature_original, feature_mixed)
        
        # Combine losses
        total_loss = loss + self.lambda_mmd * mmd_loss
        
        return total_loss

    def mmd(self, x, y):
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * xy
        
        XX = torch.zeros(xx.shape).to(x.device)
        YY = torch.zeros(xx.shape).to(x.device)
        XY = torch.zeros(xx.shape).to(x.device)
        
        bandwidth_range = [0.01, 0.1, 1, 10, 100]
        
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
        
        return torch.mean(XX + YY - 2. * XY)

class ModelStacking(nn.Module):
    def __init__(self, models):
        super(ModelStacking, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

class TrustScoreMasking:
    def __init__(self, k=3):
        self.k = k

    def __call__(self, output):
        values, _ = torch.topk(output, self.k)
        mask = torch.zeros_like(output)
        mask.scatter_(1, torch.topk(output, self.k)[1], values)
        return mask

class KnowledgeDistillation(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillation, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, x, y):
        teacher_output = self.teacher_model(x)
        student_output = self.student_model(x)
        
        loss_CE = F.cross_entropy(student_output, y)
        loss_KD = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        loss = self.alpha * loss_KD + (1 - self.alpha) * loss_CE
        return loss