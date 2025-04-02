import numpy as np
import torch

from utils.transforms import *
from utils.utils import *
from my_dataset import VOCdataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.fcn2 import *
from torch.amp import autocast, GradScaler
import datetime
import argparse
import time
import os


def main(args):

    device=torch.device(args.device if torch.cuda.is_available else 'cpu')
    print(f"using device: {device}")
    batch_size=args.batch_size
    num_classes=args.num_classes+1 # 背景+种类
    epochs=args.epochs

    checkpoint_dir=args.checkpoint_dir
    loss_dir='./results/loss/loss_curve.png'
    pretrained_weight_path="./results/weights/fcn_resnet50_coco.pth"

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    #数据
    train_dataset=VOCdataset(args.root_path,train=True,transforms=get_transforms(train=True))
    test_dataset=VOCdataset(args.root_path,train=False,transforms=get_transforms(train=False))

    train_dataloader=DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=train_dataset.collate_fn,
                                pin_memory=True,
                                num_workers=5)
    
    test_dataloader=DataLoader(test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               collate_fn=test_dataset.collate_fn,
                               pin_memory=True,
                               num_workers=5)
    
    model=FCN(num_classes=num_classes,pretrained_path=pretrained_weight_path)
    model=model.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': args.lr*0.1},  # backbone较小学习率
        {'params': model.classifier.parameters()},
        {'params': model.aux_classifier.parameters(), 'lr': args.lr*2}  # 辅助头较大学习率
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # 根据验证mIoU调整


    print("start training...")
    best_miou=0.0
    total_loss=[]
    start_time = time.time()
    for epoch in range(epochs):
        print(f"current epoch: {epoch+1}")

        weight_path = "./results/weights/VOC_segment_weight_of_epoch{}.pth".format(epoch + 1)
        if os.path.exists(weight_path):
            print(f"epoch {epoch+1} weight path exists,loading... ")
            model.load_state_dict(torch.load(weight_path))
        else:
            print(f"epoch {epoch+1} weight not found, training... ")
            model.train()
            running_loss=0.0
            scaler = GradScaler("cuda")
            for i,(img,mask) in enumerate(train_dataloader):
                img,mask=img.to(device),mask.to(device)

                optimizer.zero_grad()
                with autocast("cuda"):  # 自动混合精度
                    outputs=model(img)

                    loss=criterion(outputs,mask)

                    running_loss+=loss

                scaler.scale(loss).backward()  # 缩放梯度
                scaler.step(optimizer)
                scaler.update()


                print(f" epoch: {epoch+1} / {epochs} : ",f"image : {i+1} / {len(train_dataloader)} ",f" train loss : {running_loss/(i+1):.4f}")

            avg_loss = running_loss / len(train_dataloader)
            total_loss.append(avg_loss)
            print(f"Train Loss: {avg_loss:.4f}")

            torch.save(model.state_dict(), weight_path)

        #每个epoch过后评估
        current_miou = evaluate(model, test_dataloader, device=device, num_classes=num_classes,epoch=epoch+1,save_dir="./results/eval/")
        # 保存最佳模型
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(model.state_dict(), "./results/weights/best_model.pth")
            print(f"New best model saved with mIoU: {best_miou:.4f}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    
    plot_loss_curve(total_loss,loss_dir)


def evaluate(model, dataloader, device, num_classes, epoch=None, save_dir='./results/predict/'):
    print("Evaluating model...")
    model.eval()
    total_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # 转换预测结果
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            else:
                preds = torch.argmax(outputs, dim=1)
            # 更新混淆矩阵
            preds_flat = preds.view(-1).cpu()

            targets_flat = targets.view(-1).cpu()
            cm = confusion_matrix(preds_flat, targets_flat, num_classes)
            total_cm += cm

            # 每批次保存前两个样本的可视化结果
            if idx < 2:  # 每个batch保存前两个样本
                for i in range(min(2, images.size(0))):
                    plot_img_and_mask(
                        img=images[i],
                        mask=targets[i],
                        pred_mask=preds[i],
                        output_path=save_dir,
                        epoch=epoch,
                        index=idx * images.size(0) + i
                    )

    # 计算指标
    miou, iou_per_class = compute_miou(total_cm.numpy())

    print("\nEvaluation Results:")
    print(f"mIoU: {miou:.4f}")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i} IoU: {iou:.4f}")

    return miou


def parse_args():

    parser=argparse.ArgumentParser(description='pytorch fcn training')

    parser.add_argument('--epochs',default=3,help='training epochs')
    parser.add_argument('--num_classes',type=int,default=20,help='segment_classes')
    parser.add_argument('--lr',type=float,default=0.01,help='initial learning rate')
    parser.add_argument('--root_path',default='./data/VOCdevkit/VOC2012')
    parser.add_argument('--batch_size',type=int,default=4,help='batch_size')
    parser.add_argument('--device',default='cuda',help='device')
    parser.add_argument('--weight_decay',default=1e-4,type=float,help='weight_decay') # 权重衰减
    parser.add_argument('--checkpoint_dir',default='./results/weights',help='weight save path')
    parser.add_argument('--resume', default='', help='checkpoint path to resume')

    args=parser.parse_args()

    return args


if __name__ == "__main__":
    args=parse_args()
    
    '''model=FCN(21).to(args.device)
    test_dataset=VOCdataset(args.root_path,False,get_transforms(False))
    test_dataloader=DataLoader(test_dataset)

    evaluate(args,model,test_dataloader)'''


    # 训练模型
    main(args)

