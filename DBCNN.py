import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from matplotlib.colors import LinearSegmentedColormap

# -------------------- SCI-31 渐变配色（用于混淆矩阵 & t-SNE） --------------------
colors_rgb = np.array([
    [198,  91,  63],
    [222, 160, 145],
    [237, 204, 197],
    [215, 230, 234],
    [119, 165, 179],
    [ 64, 128, 148],
]) / 255.0
# 连续渐变色图（混淆矩阵用）
sci_cmap = LinearSegmentedColormap.from_list("sci31", colors_rgb)
# 前 4 个颜色作为离散调色板（4 分类，t-SNE 用）
sci_palette_4 = colors_rgb[:4].tolist()

# -------------------- CBAM 模块定义 --------------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(ch, ch // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ch // r, ch, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        avg_out = self.fc2(self.relu(self.fc1(avg)))
        max_out = self.fc2(self.relu(self.fc1(max)))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_out, avg_out], dim=1)
        y = self.conv(y)
        weight = self.sigmoid(y)
        return x * weight

class CBAM(nn.Module):
    def __init__(self, ch, r=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# -------------------- 基础残差块 + CBAM --------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_ch)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        if self.use_cbam:
            out = self.cbam(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, featA, featB):
        b, c, h, w = featA.size()
        n = h * w
        A = featA.view(b, c, n).permute(2, 0, 1)
        B = featB.view(b, c, n).permute(2, 0, 1)
        out, _ = self.mha(A, B, B)
        out = self.norm(A + self.dropout(out))
        return out.permute(1, 2, 0).view(b, c, h, w)

# -------------------- 主网络：BUALBranchCNN with CBAM --------------------
class BUALBranchCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BUALBranchCNN, self).__init__()
        # 分支1：ResNet风格 + CBAM
        self.branch1 = nn.Sequential(
            self._make_layer(3, 32, blocks=2, stride=1), nn.MaxPool2d(2),
            self._make_layer(32, 64, blocks=2, stride=1), nn.MaxPool2d(2),
            self._make_layer(64, 128, blocks=2, stride=1, use_cbam=True), nn.MaxPool2d(2)
        )
        # 分支2：不同感受野 + Residual + CBAM
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False), nn.ReLU(inplace=True),
            self._make_layer(32, 64, blocks=2, stride=1), nn.MaxPool2d(2),
            self._make_layer(64, 128, blocks=2, stride=1, use_cbam=True), nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )
        self.cross_attn1 = CrossAttention(embed_dim=128, num_heads=4)
        self.cross_attn2 = CrossAttention(embed_dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1, use_cbam=False):
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
        ) if (in_ch != out_ch or stride != 1) else None
        layers = [BasicBlock(in_ch, out_ch, stride, downsample, use_cbam)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, use_cbam=use_cbam))
        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f1_cross = self.cross_attn1(f1, f2)
        f2_cross = self.cross_attn2(f2, f1)
        fused = torch.cat([f1_cross, f2_cross], dim=1)
        return self.classifier(fused)

# ------------------------------ 主程序 ------------------------------
if __name__ == "__main__":
    start_time = time.time()
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集

    class AddGaussianNoise(object):
        """
        给 Tensor 图像添加指定 SNR 的高斯白噪声。
        SNR_db 为目标信噪比（dB），
        计算：
            P_signal = E[x^2]
            P_noise  = P_signal / (10^(SNR_db/10))
            noise_std = sqrt(P_noise)
        """

        def __init__(self, snr_db=10.0):
            self.snr_db = snr_db

        def __call__(self, tensor):
            # tensor: C×H×W，范围假设已在 [0,1] 或者标准化之后
            # 1) 计算信号功率
            p_signal = tensor.pow(2).mean()
            # 2) 根据信噪比计算噪声功率
            p_noise = p_signal / (10 ** (self.snr_db / 10))
            # 3) 生成噪声
            noise = torch.randn_like(tensor) * torch.sqrt(p_noise)
            return tensor + noise

        def __repr__(self):
            return f"{self.__class__.__name__}(snr_db={self.snr_db})"

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        AddGaussianNoise(snr_db=3),  # ← 在这里添加噪声
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集


    data_dir = r'F:\pyproject\DBCNN\mohubupinghua_fig_4A_700hz'

    print("数据目录地址：", data_dir)

    num_epochs = 20
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # 按类别划分索引
    classes = full_dataset.classes
    num_classes = len(classes)
    idxs_per_class = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(full_dataset.samples):
        idxs_per_class[label].append(idx)

    test_per_class = test_size // num_classes
    val_per_class = val_size // num_classes
    generator = torch.Generator().manual_seed(42)
    train_idxs, val_idxs, test_idxs = [], [], []
    for cls in range(num_classes):
        all_idxs = torch.tensor(idxs_per_class[cls])
        shuffled = all_idxs[torch.randperm(len(all_idxs), generator=generator)].tolist()
        test_chunk = shuffled[:test_per_class]
        val_chunk = shuffled[test_per_class:test_per_class + val_per_class]
        train_chunk = shuffled[test_per_class + val_per_class:]
        test_idxs.extend(test_chunk)
        val_idxs.extend(val_chunk)
        train_idxs.extend(train_chunk)

    train_loader = DataLoader(Subset(full_dataset, train_idxs), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idxs), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idxs), batch_size=32, shuffle=False)

    # 模型、损失与优化器
    model = BUALBranchCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 训练与验证循环
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(100. * correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        # 在训练循环开始前定义
        best_val_acc = 0.0

        # 训练与验证循环
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_losses.append(total_loss / len(train_loader))
            train_accs.append(100. * correct / total)

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()

                    preds = outputs.argmax(dim=1)
                    val_correct += preds.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_losses.append(val_loss / len(val_loader))
            val_accs.append(100. * val_correct / val_total)

            # 保存验证集准确率最高的模型
            current_val_acc = val_accs[-1]

            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "classes": classes,
                    "class_to_idx": full_dataset.class_to_idx,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch + 1,
                }, "best_model.pth")

                print(f"Best model saved at epoch {epoch + 1}, val acc = {best_val_acc:.2f}%")

            scheduler.step()

            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.2f}% | "
                  f"Val Loss {val_losses[-1]:.4f}, Acc {val_accs[-1]:.2f}%")

    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_curves.png', dpi=600)

    # 测试与评估
    all_labels, all_preds, all_feats = [], [], []
    test_loss, test_correct, test_total = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_feats.extend(outputs.cpu().tolist())

    test_acc = 100. * test_correct / test_total
    print(f"Test Loss {test_loss/len(test_loader):.4f}, Acc {test_acc:.2f}%")
    # 创建一个 DataFrame 来保存训练和验证的损失和准确率
    training_data = {
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accs,
        'val_accuracy': val_accs
    }

    df = pd.DataFrame(training_data)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv('training_history.csv', index=False)
    print("Training and validation history saved to training_history.csv")

    # 混淆矩阵
    # 混淆矩阵可视化并保存

    # 将 all_preds 转换为 DataFrame
    df = pd.DataFrame(all_preds, columns=['Predictions'])
    # 保存为 CSV 文件
    df.to_csv('predictions_wotaihua.csv', index=False)
    file_path = 'F:\\pyproject\\DBCNN\\predictions_wotaihua.csv'  # 替换为你的文件路径
    #all_preds = pd.read_csv(file_path)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    annot_kws = {"size": 33}
    # 画热力图（使用 SCI-31 渐变配色）
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=sci_cmap,
        xticklabels=classes,
        yticklabels=classes,
        annot_kws=annot_kws,
        cbar=True,
        square=True,
        cbar_kws={"shrink": 0.8, "aspect": 5}
    )
    # 设置坐标轴字体
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 获取 colorbar 对象
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    # 设置 colorbar 刻度，包含最大值 156
    max_val = 156
    cbar.set_ticks([0, 30, 60, 90, 120, max_val])
    cbar.set_ticklabels([0, 30, 60, 90, 120, max_val])
    plt.tight_layout()
    plt.savefig('confusion_matrix_wotaihuau.png', dpi=600)
    print("Confusion matrix saved with max colorbar label.")

    # t-SNE 可视化并保存
    features_np = np.array(all_feats)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_feats = tsne.fit_transform(features_np)

    df = pd.DataFrame(
        {'tsne-2d-1': tsne_feats[:, 0],
         'tsne-2d-2': tsne_feats[:, 1],
         'label': [classes[i] for i in all_labels]}
    )

    plt.figure(figsize=(6, 5))
    sci_palette_4 = {
        'Critical': '#D55E00',  # orange-red
        'Health': '#009E73',  # green
        'Moderate': '#0072B2',  # blue
        'Slight': '#E69F00'  # orange
    }
    ax = sns.scatterplot(
        data=df,
        x='tsne-2d-1',
        y='tsne-2d-2',
        hue='label',
        palette=sci_palette_4,
        s=45,
        alpha=0.9,
        linewidth=0.15,
        edgecolor='white',
        legend=False
    )

    # 去掉横纵坐标标题
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 去掉刻度和刻度标签
    ax.set_xticks([])
    ax.set_yticks([])

    # 保留外边框，让子图边界清楚
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=600, bbox_inches='tight')
    plt.show()

# 测试与评估
all_labels, all_preds = [], []
test_loss, test_correct, test_total = 0, 0, 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        preds = outputs.argmax(dim=1)
        test_correct += preds.eq(labels).sum().item()
        test_total += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

test_acc = 100. * test_correct / test_total
print(f"Test Loss {test_loss/len(test_loader):.4f}, Acc {test_acc:.2f}%")

#all_preds = all_labels
# 计算各项指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
# 创建一个 DataFrame 来保存训练和验证的损失和准确率
training_data = {
    'epoch': list(range(1, num_epochs + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accs,
    'val_accuracy': val_accs
}

df = pd.DataFrame(training_data)

# 将 DataFrame 保存为 CSV 文件
df.to_csv('fgprp_training_history.csv', index=False)
print("Training and validation history saved to training_history.csv")
plt.show()
end_time = time.time()  # ← 添加这行
total_time_seconds = end_time - start_time
print(f"\n🕒 总运行时间：{total_time_seconds:.2f} 秒 ≈ {total_time_seconds / 60:.2f} 分钟")


