
# utils/visualization.py

import visdom
import numpy as np
from config import VISDOM_SERVER, VISDOM_PORT

def setup_visdom():
    return visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT)

def plot_loss(vis, epoch, loss, win='loss'):
    vis.line(X=[epoch], Y=[loss], win=win, update='append' if epoch > 0 else None, opts=dict(title='Training Loss'))

def plot_accuracy(vis, epoch, accuracy, win='accuracy'):
    vis.line(X=[epoch], Y=[accuracy], win=win, update='append' if epoch > 0 else None, opts=dict(title='Accuracy'))

def update_single_plot(vis, Y, win, title, ylabel, defense_methods):
    vis.line(
        X=np.arange(len(Y)),
        Y=np.array(Y),
        win=win,
        opts=dict(
            title=title,
            xlabel='Defense Methods',
            ylabel=ylabel,
            xtick=True,
            xtickvals=list(range(len(defense_methods))),
            xlabels=defense_methods,
            legend=['Attack Accuracy'],
            markersymbol='dot',
            markersize=10,
        ),
    )

def update_multi_line_plot(vis, Y, win, title, legend_labels, defense_methods):
    vis.line(
        X=np.tile(np.arange(len(Y)), (len(legend_labels), 1)).T,
        Y=np.array(Y),
        win=win,
        opts=dict(
            title=title,
            xlabel='Defense Methods',
            ylabel='Score',
            xtick=True,
            xtickvals=list(range(len(defense_methods))),
            xlabels=defense_methods,
            legend=legend_labels,
            markersymbol='dot',
            markersize=10,
        ),
    )

def update_visdom_plots(vis, attack_accuracies, macro_avgs, weighted_avgs):
    # 定义固定的防御方法列表
    defense_methods = [
        'L2Regularization', 'DropoutDefense', 'LabelSmoothing', 'AdversarialRegularization',
        'MixupMMD', 'ModelStacking', 'TrustScoreMasking', 'KnowledgeDistillation'
    ]

    if vis:
        # 窗口 1: 攻击准确率
        update_single_plot(vis, attack_accuracies, 'attack_accuracy_plot', 
                           'Accuracy of Member Inference Attacks under Different Defense Methods', 'Attack accuracy rate', defense_methods)
        
        # 窗口 2: 宏平均
        update_multi_line_plot(vis, macro_avgs, 'macro_avg_plot', 
                               'Macro average performance indicator', ['precision', 'recall', 'F1-score'], defense_methods)
        
        # 窗口 3: 加权平均
        update_multi_line_plot(vis, weighted_avgs, 'weighted_avg_plot', 
                               'Weighted average performance indicators', ['precision', 'recall', 'F1-score'], defense_methods)
    else:
        print("Visdom object is not available. Skipping visualization.")