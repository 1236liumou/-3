"""
SVM 字符识别模型训练脚本

功能:
  1. 加载 prepare_ccpd.py 生成的训练数据
  2. 提取 HOG + 像素特征
  3. 训练 SVM 分类器
  4. 在验证集上评估
  5. 保存模型

使用方式:
    python scripts/train_svm.py --data_dir data/ --output models/svm_model.pkl

前置条件:
    先运行 prepare_ccpd.py 生成训练数据
"""

import os
import cv2
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.config import CHAR_RECOGNITION_CONFIG, DATA_DIR
from lpr.char_recognition import build_char_to_label, build_char_map, NUM_CLASSES


def load_dataset(data_dir):
    """
    加载训练数据
    目录结构: data_dir/<char>/<idx_pos>.png

    返回:
        images: 图像列表
        labels: 标签列表
        char_set: 出现过的字符集合
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    char_to_label = build_char_to_label()
    char_counts = defaultdict(int)

    # 遍历所有子目录 (每个目录名是一个字符)
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

            # 二值化
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # 标准化到 20x28
            img = cv2.resize(img, (20, 28), interpolation=cv2.INTER_CUBIC)

            images.append(img)
            labels.append(label)
            char_counts[char] += 1

    print(f"  加载 {len(images)} 个样本，{len(char_counts)} 种字符")
    return images, labels, char_counts


def extract_hog_features(image):
    """
    提取 HOG + 像素特征
    HOG: 方向梯度直方图，捕捉字符形状信息
    像素: 降采样后的像素值，补充全局信息
    """
    # 确保是灰度图且尺寸正确
    if image.shape != (28, 20):
        image = cv2.resize(image, (20, 28))

    # HOG 特征
    hog = cv2.HOGDescriptor(
        _winSize=(20, 28),
        _blockSize=(10, 14),
        _blockStride=(5, 7),
        _cellSize=(5, 7),
        _nbins=9,
    )
    hog_features = hog.compute(image).flatten()

    # 像素特征 (降采样到 10x14)
    small = cv2.resize(image, (10, 14)).flatten()
    pixel_features = small.astype(np.float32) / 255.0

    # 拼接特征
    features = np.concatenate([hog_features, pixel_features])
    return features


def extract_features_batch(images):
    """批量提取特征"""
    features = []
    for img in images:
        feat = extract_hog_features(img)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def train_svm(data_dir, output_path, C=10.0, kernel="rbf", gamma="scale"):
    """
    训练 SVM 模型

    参数:
        data_dir: 数据目录 (含 train/ 和 val/ 子目录)
        output_path: 模型保存路径
        C: 正则化参数
        kernel: 核函数 ("rbf" | "linear" | "poly")
        gamma: 核系数 ("scale" | "auto" 或浮点数)
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import json

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
    print("SVM 字符识别模型训练")
    print("=" * 50)

    print("\n[1/5] 加载训练数据...")
    train_images, train_labels, train_chars = load_dataset(train_dir)
    print(f"  训练集: {len(train_images)} 个样本")

    val_images, val_labels, val_chars = [], [], {}
    if val_dir.exists():
        val_images, val_labels, val_chars = load_dataset(val_dir)
        print(f"  验证集: {len(val_images)} 个样本")

    # ============================================================
    # 2. 特征提取
    # ============================================================
    print("\n[2/5] 提取特征...")
    print("  提取训练集特征...")
    X_train = extract_features_batch(train_images)
    y_train = np.array(train_labels)
    print(f"  特征维度: {X_train.shape}")

    if val_images:
        print("  提取验证集特征...")
        X_val = extract_features_batch(val_images)
        y_val = np.array(val_labels)

    # ============================================================
    # 3. 特征标准化
    # ============================================================
    print("\n[3/5] 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if val_images:
        X_val_scaled = scaler.transform(X_val)

    # ============================================================
    # 4. 训练 SVM
    # ============================================================
    print(f"\n[4/5] 训练 SVM (C={C}, kernel={kernel}, gamma={gamma})...")

    # 使用概率估计 (需要 probability=True)
    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        decision_function_shape="ovr",
        verbose=True,
    )
    svm.fit(X_train_scaled, y_train)
    print("  训练完成!")

    # 训练集准确率
    train_pred = svm.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"  训练集准确率: {train_acc:.4f}")

    # 验证集准确率
    if val_images:
        val_pred = svm.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  验证集准确率: {val_acc:.4f}")

    # ============================================================
    # 5. 保存模型
    # ============================================================
    print(f"\n[5/5] 保存模型到 {output_path}...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model_data = {
        "svm": svm,
        "scaler": scaler,
        "char_map": build_char_map(),
        "char_to_label": build_char_to_label(),
        "config": {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
            "feature_type": "hog+pixel",
            "input_size": (20, 28),
        },
        "train_accuracy": train_acc,
        "val_accuracy": val_acc if val_images else None,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存: {output_path}")
    print(f"  训练集准确率: {train_acc:.4f}")
    if val_images:
        print(f"  验证集准确率: {val_acc:.4f}")

    # ============================================================
    # 分类报告 (验证集)
    # ============================================================
    if val_images:
        print("\n验证集分类报告 (前20类):")
        char_map = build_char_map()
        target_names = [char_map.get(i, str(i)) for i in range(NUM_CLASSES)]
        report = classification_report(
            y_val, val_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        # 只打印前20个字符的报告
        printed = 0
        for char_name in target_names:
            if char_name in report:
                r = report[char_name]
                print(f"  {char_name}: precision={r['precision']:.2f} "
                      f"recall={r['recall']:.2f} f1={r['f1-score']:.2f} "
                      f"support={r['support']}")
                printed += 1
                if printed >= 20:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM 字符识别训练")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="数据目录 (含 train/ 和 val/)")
    parser.add_argument("--output", type=str,
                        default="models/svm_model.pkl",
                        help="模型保存路径")
    parser.add_argument("--C", type=float, default=10.0,
                        help="正则化参数")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["rbf", "linear", "poly"],
                        help="核函数")
    parser.add_argument("--gamma", type=str, default="scale",
                        help="核系数")

    args = parser.parse_args()

    train_svm(
        data_dir=args.data_dir,
        output_path=args.output,
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
    )
