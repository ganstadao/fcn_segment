import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def plot_loss_curve(loss,path:str):
    plt.plot(loss)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(path)
    plt.show()

def confusion_matrix(y_pred,y_true,num_classes):
    """计算混淆矩阵"""
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = torch.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return cm


def plot_img_and_mask(img, mask, pred_mask, output_path, epoch=None, index=0):
    """
    绘制原始图像、真实mask和预测mask的对比图
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像（需要反标准化）
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    ax[0].imshow(img)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    # 真实mask
    if classes > 1:
        true_mask = torch.argmax(mask, dim=0).cpu().numpy()
    else:
        true_mask = mask.cpu().numpy().squeeze()

    # 剔除255像素值（忽略区域）
    true_mask = np.where(true_mask == 255, 0, true_mask)  # 将255替换为0（背景）

    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    # 预测mask
    if classes > 1:
        pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()
    else:
        pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    ax[2].imshow(pred_mask.squeeze(), cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    # 保存图像
    if epoch is not None:
        output_path = os.path.join(output_path, f'epoch_{epoch}_sample_{index}.png')
    else:
        output_path = os.path.join(output_path, f'sample_{index}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def compute_miou(cm):
    """从混淆矩阵计算mIoU"""
    iou_per_class = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[i, :].sum() - tp
        fn = cm[:, i].sum() - tp
        denominator = tp + fp + fn
        if denominator == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(tp / denominator)
    miou = np.nanmean(iou_per_class)
    return miou, iou_per_class

