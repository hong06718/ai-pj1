"""
街景字符识别 - 改进版
复旦大学人工智能课程 随堂练习3

相比 baseline 的改进：
1. 更强的骨干网络：EfficientNet-B4（通过 timm），fallback 用 ResNet101
2. 修复 LabelSmoothEntropy 中 shape[0]→shape[1] 的 Bug
3. 修复 eval() 精度计算 Bug（用实际样本数而非 batch_size 估算）
4. 混合精度训练（AMP）加速，显存占用减半
5. 更强数据增强：RandomPerspective、RandomErasing、更强 ColorJitter
6. OneCycleLR（10% warmup + 余弦退火）
7. Dropout + 梯度裁剪防过拟合
8. 训练轮数 20 轮（baseline 仅 2 轮）
"""

import os
import json
import random
import zipfile
import requests
from glob import glob

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# ── 骨干网络选择 ───────────────────────────────────────────────────────────────
try:
    import timm
    USE_TIMM = True
    print("✓ timm 可用，将使用 EfficientNet-B4 骨干网络")
except ImportError:
    USE_TIMM = False
    print("⚠ timm 未安装，将使用 ResNet101。安装 timm 可获得更好效果：pip install timm")

# ── 下载 & 解压数据集 ──────────────────────────────────────────────────────────
links = pd.read_csv('./mchar_data_list_0515.csv')
dataset_path = "./dataset"
print(f"数据集目录：{dataset_path}")
os.makedirs(dataset_path, exist_ok=True)

for i, link in enumerate(links['link']):
    file_name = links['file'][i]
    full_path = os.path.join(dataset_path, file_name)
    if not os.path.exists(full_path):
        print(f"下载 {file_name} ...")
        response = requests.get(link, stream=True)
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

for little_zip in ['mchar_train', 'mchar_test_a', 'mchar_val']:
    zip_name = os.path.join(dataset_path, little_zip)
    if not os.path.exists(zip_name):
        zf = zipfile.ZipFile(os.path.join(dataset_path, f"{little_zip}.zip"), 'r')
        zf.extractall(path=dataset_path)

# ── 数据路径 ───────────────────────────────────────────────────────────────────
data_dir = {
    'train_data':  f'{dataset_path}/mchar_train/',
    'val_data':    f'{dataset_path}/mchar_val/',
    'test_data':   f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label':   f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv',
}

print('train: %d  val: %d  test: %d' % (
    len(glob(data_dir['train_data'] + '*.png')),
    len(glob(data_dir['val_data']   + '*.png')),
    len(glob(data_dir['test_data']  + '*.png')),
))

# ── 超参数 ─────────────────────────────────────────────────────────────────────
class Config:
    batch_size   = 64
    lr           = 1e-3
    weight_decay = 1e-4
    class_num    = 11
    epoches      = 20        # baseline 只有 2 轮，这里改为 20 轮
    smooth       = 0.1
    erase_prob   = 0.5
    input_h      = 128
    input_w      = 224
    num_workers  = 4         # Windows 下建议 4，Linux 下可用 8
    checkpoints  = './checkpoints'
    pretrained   = None
    start_epoch  = 0
    eval_interval = 1

config = Config()

# ── Dataset ────────────────────────────────────────────────────────────────────
class DigitsDataset(Dataset):
    def __init__(self, mode='train', aug=True):
        super().__init__()
        self.aug   = aug
        self.mode  = mode
        self.width = config.input_w
        self.batch_count = 0

        if mode == 'test':
            self.imgs   = sorted(glob(data_dir['test_data'] + '*.png'))
            self.labels = None
        else:
            labels = json.load(open(data_dir[f'{mode}_label'], 'r'))
            imgs   = glob(data_dir[f'{mode}_data'] + '*.png')
            self.imgs = [
                (img, labels[os.path.basename(img)])
                for img in imgs
                if os.path.basename(img) in labels
            ]

    # 构造 transforms
    def _build_transforms(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
        base = [
            transforms.Resize(config.input_h),
            transforms.CenterCrop((config.input_h, self.width)),
        ]
        if self.aug:
            aug_list = [
                # 颜色扰动：亮度/对比度/饱和度/色调
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                # 几何变换：旋转 + 平移 + 剪切
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.1), shear=5),
                # 透视变换：模拟不同拍摄角度
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            ]
            base.extend(aug_list)

        base += [transforms.ToTensor(), normalize]

        if self.aug:
            # RandomErasing 必须在 ToTensor 之后
            base.append(transforms.RandomErasing(
                p=config.erase_prob, scale=(0.02, 0.15)
            ))

        return transforms.Compose(base)

    def __getitem__(self, idx):
        trans = self._build_transforms()

        if self.mode != 'test':
            img_path, label = self.imgs[idx]
            img = Image.open(img_path).convert('RGB')
            label_tensor = t.tensor(
                label['label'][:4] + (4 - len(label['label'])) * [10]
            ).long()
            return trans(img), label_tensor
        else:
            img_path = self.imgs[idx]
            img = Image.open(img_path).convert('RGB')
            return trans(img), img_path

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))
        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels)

# ── 模型 ───────────────────────────────────────────────────────────────────────
class DigitsNet(nn.Module):
    """
    多头分类网络：对门牌号前4位分别做 11分类（0-9 + 空）
    骨干网络优先使用 EfficientNet-B4（timm），否则使用 ResNet101
    """
    def __init__(self, class_num=11):
        super().__init__()
        if USE_TIMM:
            # num_classes=0 让 timm 去掉分类头，只输出特征向量
            self.backbone = timm.create_model(
                'efficientnet_b4', pretrained=True, num_classes=0
            )
            feat_dim = self.backbone.num_features  # 1792
        else:
            from torchvision.models.resnet import resnet101
            net = resnet101(pretrained=True)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            feat_dim = 2048

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(feat_dim, class_num)
        self.fc2 = nn.Linear(feat_dim, class_num)
        self.fc3 = nn.Linear(feat_dim, class_num)
        self.fc4 = nn.Linear(feat_dim, class_num)

    def forward(self, img):
        feat = self.backbone(img)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(feat)
        return self.fc1(feat), self.fc2(feat), self.fc3(feat), self.fc4(feat)

# ── 损失函数 ───────────────────────────────────────────────────────────────────
class LabelSmoothEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    修复了 baseline 中 preds.shape[0]（batch size）应为 preds.shape[1]（类别数）的 Bug
    """
    def __init__(self, smooth=0.1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        n_class  = preds.shape[1]                        # 修复：用类别数而非 batch size
        lb_pos   = 1.0 - self.smooth
        lb_neg   = self.smooth / (n_class - 1)
        smoothed = t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        log_prob = F.log_softmax(preds, dim=1)
        return -(log_prob * smoothed).sum(1).mean()

# ── Trainer ────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, val=True):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        print(f"训练设备：{self.device}")

        # AMP 梯度缩放器（CPU 时自动禁用）
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')

        # 数据集
        self.train_set    = DigitsDataset(mode='train', aug=True)
        self.train_loader = DataLoader(
            self.train_set, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            pin_memory=True, persistent_workers=(config.num_workers > 0),
            drop_last=True, collate_fn=self.train_set.collect_fn,
        )

        if val:
            val_set = DigitsDataset(mode='val', aug=False)
            self.val_loader = DataLoader(
                val_set, batch_size=config.batch_size,
                num_workers=config.num_workers, pin_memory=True,
                persistent_workers=(config.num_workers > 0), drop_last=False,
            )
            self.val_total = len(val_set)
        else:
            self.val_loader = None
            self.val_total  = 0

        # 模型、损失、优化器
        self.model     = DigitsNet(config.class_num).to(self.device)
        self.criterion = LabelSmoothEntropy(smooth=config.smooth).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # OneCycleLR：前 10% 线性 warmup，之后余弦退火
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            steps_per_epoch=len(self.train_loader),
            epochs=config.epoches,
            pct_start=0.1,
        )

        self.best_acc            = 0.0
        self.best_checkpoint_path = ""

        if config.pretrained is not None:
            self.load_model(config.pretrained)
            acc = self.eval()
            self.best_acc = acc
            print(f"加载预训练模型 {config.pretrained}，验证集 Acc: {acc:.4f}")

    # ── 完整训练 ──────────────────────────────────────────────────────────────
    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            self.train_epoch(epoch)
            if (epoch + 1) % config.eval_interval == 0:
                acc = self.eval()
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(
                        config.checkpoints,
                        f'epoch-{epoch+1}-acc-{acc:.4f}.pth'
                    )
                    self.save_model(save_path)
                    print(f"  ✓ 保存最优模型 → {save_path}")
                    self.best_acc            = acc
                    self.best_checkpoint_path = save_path
        print(f"\n训练完成，最优验证集 Acc: {self.best_acc:.4f}")

    # ── 单轮训练 ──────────────────────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        corrects   = 0
        total      = 0
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.epoches}')

        for i, (img, label) in enumerate(tbar):
            img   = img.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.device.type == 'cuda'):
                pred = self.model(img)
                loss = (
                    self.criterion(pred[0], label[:, 0]) +
                    self.criterion(pred[1], label[:, 1]) +
                    self.criterion(pred[2], label[:, 2]) +
                    self.criterion(pred[3], label[:, 3])
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            total_loss += loss.item()
            bs          = img.size(0)
            total      += bs
            correct_mask = t.stack([
                pred[0].argmax(1) == label[:, 0],
                pred[1].argmax(1) == label[:, 1],
                pred[2].argmax(1) == label[:, 2],
                pred[3].argmax(1) == label[:, 3],
            ], dim=1)
            corrects += t.all(correct_mask, dim=1).sum().item()

            tbar.set_description(
                f'Epoch {epoch+1} | loss: {total_loss/(i+1):.3f} '
                f'| train acc: {corrects*100/total:.2f}%'
            )

    # ── 验证 ──────────────────────────────────────────────────────────────────
    def eval(self):
        self.model.eval()
        corrects = 0
        total    = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader, desc='验证中')
            for img, label in tbar:
                img   = img.to(self.device)
                label = label.to(self.device)
                with autocast(enabled=self.device.type == 'cuda'):
                    pred = self.model(img)
                correct_mask = t.stack([
                    pred[0].argmax(1) == label[:, 0],
                    pred[1].argmax(1) == label[:, 1],
                    pred[2].argmax(1) == label[:, 2],
                    pred[3].argmax(1) == label[:, 3],
                ], dim=1)
                corrects += t.all(correct_mask, dim=1).sum().item()
                total    += img.size(0)
                tbar.set_description(f'Val Acc: {corrects*100/total:.2f}%')

        acc = corrects / total
        print(f'\n验证集 Acc: {acc:.4f}  ({corrects}/{total})\n')
        self.model.train()
        return acc

    def save_model(self, path):
        t.save({'model': self.model.state_dict()}, path)

    def load_model(self, path):
        self.model.load_state_dict(t.load(path, map_location=self.device)['model'])

# ── 推理 & 生成提交文件 ────────────────────────────────────────────────────────
def parse2class(prediction):
    char_list = [str(i) for i in range(10)] + ['']
    ch = [pred.argmax(1) for pred in prediction]
    return [
        ''.join(char_list[ch[j][k].item()] for j in range(4))
        for k in range(len(ch[0]))
    ]

def write2csv(results, csv_path):
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(
        lambda x: os.path.basename(x)
    )
    df.to_csv(csv_path, sep=',', index=None)
    print(f'结果已保存至 {csv_path}')

def predicts(model_path, csv_path):
    device   = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    test_set = DigitsDataset(mode='test', aug=False)
    loader   = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=(config.num_workers > 0), drop_last=False,
    )
    model = DigitsNet(config.class_num).to(device)
    model.load_state_dict(t.load(model_path, map_location=device)['model'])
    model.eval()
    print(f'加载模型 {model_path} 成功')

    results = []
    with t.no_grad():
        for img, img_names in tqdm(loader, desc='预测中'):
            img = img.to(device)
            with autocast(enabled=device.type == 'cuda'):
                pred = model(img)
            results += [
                [name, code]
                for name, code in zip(img_names, parse2class(pred))
            ]

    results = sorted(results, key=lambda x: os.path.basename(x[0]))
    write2csv(results, csv_path)
    return results

# ── 主程序 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    trainer = Trainer(val=True)
    trainer.train()
    predicts(trainer.best_checkpoint_path, "result.csv")
