"""
CNN 字符识别模型训练脚本

功能:
  1. 加载 prepare_ccpd.py 生成的训练数据
  2. 数据增强 (旋转、平移、缩放、噪声)
  3. 训练 CNN 分类器 (SimpleCharCNN)
  4. 在验证集上评估
  5. 保存模型

使用方式:
    python scripts/train_cnn.py --data_dir data/ --output models/cnn_model.pth --epochs 30

前置条件:
    1. 安装 PyTorch: pip install torch torchvision
    2. 先运行 prepare_ccpd.py 生成训练数据
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.config import CHAR_RECOGNITION_CONFIG, DATA_DIR
from lpr.char_recognition import build_char_to_label, build_char_map, NUM_CLASSES, SimpleCharCNN


def load_dataset(data_dir):
    """
    加载训练数据
    返回: (images, labels)
    images: numpy array (N, 1, 28, 20) float32, 归一化到 [0, 1]
    labels: numpy array (N,) int64
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    char_to_label = build_char_to_label()
    char_counts = defaultdict(int)

    for char_dir in sorted(data_path.iterdir()):
        if not char_dir.is_dir():
            continue

        char = char_dir.name
        if char not in char_to_label:
            continue

        label = char_to_label[char]

        for img_file in char_dir.glob("*.png"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 标准化到 20x28
            img = cv2.resize(img, (20, 28), interpolation=cv2.INTER_CUBIC)

            # 二值化
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)
            char_counts[char] += 1

    images = np.array(images, dtype=np.float32)  # (N, 28, 20)
    labels = np.array(labels, dtype=np.int64)

    # 添加通道维度 (N, 1, 28, 20)
    images = images[:, np.newaxis, :, :]

    print(f"  加载 {len(images)} 个样本，{len(char_counts)} 种字符")
    return images, labels, char_counts


class DataAugmentor:
    """数据增强"""

    def __init__(self, rotation_range=10, shift_range=0.1,
                 zoom_range=0.1, noise_std=0.05):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.noise_std = noise_std

    def augment(self, image):
        """
        对单张图像进行随机增强
        image: (1, 28, 20) numpy array
        """
        img = image[0].copy()  # (28, 20)

        h, w = img.shape

        # 1. 随机旋转
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

        # 2. 随机平移
        dx = random.uniform(-self.shift_range, self.shift_range) * w
        dy = random.uniform(-self.shift_range, self.shift_range) * h
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

        # 3. 随机缩放
        scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        # 裁剪/填充回原始尺寸
        if img.shape[0] > h or img.shape[1] > w:
            # 中心裁剪
            y_start = (img.shape[0] - h) // 2
            x_start = (img.shape[1] - w) // 2
            img = img[y_start:y_start + h, x_start:x_start + w]
        elif img.shape[0] < h or img.shape[1] < w:
            canvas = np.zeros((h, w), dtype=np.float32)
            y_start = (h - img.shape[0]) // 2
            x_start = (w - img.shape[1]) // 2
            canvas[y_start:y_start + img.shape[0],
                   x_start:x_start + img.shape[1]] = img
            img = canvas

        # 4. 随机高斯噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img[np.newaxis, :, :]  # (1, 28, 20)


def train_cnn(data_dir, output_path, epochs=30, batch_size=64, lr=0.001,
              use_augmentation=True, use_gpu=False):
    """
    训练 CNN 模型

    参数:
        data_dir: 数据目录
        output_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        use_augmentation: 是否使用数据增强
        use_gpu: 是否使用 GPU
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    if not train_dir.exists():
        print(f"错误: 训练数据目录不存在: {train_dir}")
        print("请先运行: python scripts/prepare_ccpd.py")
        return

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("=" * 50)
    print("CNN 字符识别模型训练")
    print("=" * 50)

    print("\n[1/6] 加载数据...")
    X_train, y_train, train_chars = load_dataset(train_dir)
    print(f"  训练集: {X_train.shape}")

    X_val, y_val = [], []
    if val_dir.exists():
        X_val, y_val, _ = load_dataset(val_dir)
        print(f"  验证集: {X_val.shape}")

    # ============================================================
    # 2. 数据增强
    # ============================================================
    if use_augmentation:
        print("\n[2/6] 数据增强...")
        augmentor = DataAugmentor()
        augmented_images = []
        augmented_labels = []

        # 对每个样本生成一个增强版本 (数据量翻倍)
        for i in range(len(X_train)):
            augmented_images.append(augmentor.augment(X_train[i]))
            augmented_labels.append(y_train[i])

        X_aug = np.array(augmented_images, dtype=np.float32)
        y_aug = np.array(augmented_labels, dtype=np.int64)

        # 合并原始 + 增强
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

        # 打乱顺序
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        print(f"  增强后训练集: {X_train.shape}")
    else:
        print("\n[2/6] 跳过数据增强")

    # ============================================================
    # 3. 转为 PyTorch 数据集
    # ============================================================
    print("\n[3/6] 准备数据加载器...")

    # 将单通道转为三通道 (网络输入 3 通道)
    X_train_3ch = np.repeat(X_train, 3, axis=1)  # (N, 3, 28, 20)
    train_tensor_x = torch.FloatTensor(X_train_3ch)
    train_tensor_y = torch.LongTensor(y_train)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    if len(X_val) > 0:
        X_val_3ch = np.repeat(X_val, 3, axis=1)
        val_tensor_x = torch.FloatTensor(X_val_3ch)
        val_tensor_y = torch.LongTensor(y_val)
        val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
    else:
        val_loader = None

    # ============================================================
    # 4. 构建模型
    # ============================================================
    print("\n[4/6] 构建模型...")
    model = SimpleCharCNN(num_classes=NUM_CLASSES).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ============================================================
    # 5. 训练循环
    # ============================================================
    print(f"\n[5/6] 开始训练 (epochs={epochs}, batch_size={batch_size}, lr={lr})...")

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            val_acc = val_correct / val_total

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        scheduler.step()

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}, lr={lr_current:.6f}")

    print(f"\n  最佳验证准确率: {best_val_acc:.4f}")

    # ============================================================
    # 6. 保存模型
    # ============================================================
    print(f"\n[6/6] 保存模型到 {output_path}...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 保存最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), output_path)

    print(f"\n模型已保存: {output_path}")
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    print(f"  模型参数量: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN 字符识别训练")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="数据目录 (含 train/ 和 val/)")
    parser.add_argument("--output", type=str,
                        default="models/cnn_model.pth",
                        help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="禁用数据增强")
    parser.add_argument("--gpu", action="store_true",
                        help="使用 GPU 训练")

    args = parser.parse_args()

    train_cnn(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=not args.no_augmentation,
        use_gpu=args.gpu,
    )
