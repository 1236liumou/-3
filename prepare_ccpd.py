"""
CCPD 数据集准备脚本

功能:
  1. 解析 CCPD 文件名，提取标注信息
  2. 裁剪车牌区域并保存
  3. 分割字符并保存为训练数据
  4. 自动划分训练集 / 验证集 / 测试集

使用方式:
    python scripts/prepare_ccpd.py --ccpd_dir /path/to/CCPD --output_dir data/

CCPD 文件名格式:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    面积比-倾斜度_边界框-四角点-车牌号索引-亮度-模糊度
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import random

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.utils import parse_ccpd_filename
from lpr.char_segmentation import CharSegmenter
from lpr.config import CHAR_RECOGNITION_CONFIG


def categorize_plate(plate_text):
    """
    根据车牌号判断车牌类型
    返回: "blue" | "green" | "yellow"
    """
    # 新能源绿牌: 长度8位，第3位是 D 或 F
    if len(plate_text) == 8 and plate_text[2] in ("D", "F"):
        return "green"
    # 黄牌: 大型车/教练车/挂车 (通常长度也是7位，但颜色不同)
    # CCPD 数据集中黄牌单独有子集
    return "blue"


def crop_plate_from_ccpd(image, annotation):
    """
    使用 CCPD 标注的四角点裁剪车牌区域
    进行透视变换校正
    """
    corners = np.array(annotation["corners"], dtype="float32")

    # 排序四角点: 左上、右上、右下、左下
    s = corners.sum(axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = corners[np.argmin(s)]  # 左上
    ordered[2] = corners[np.argmax(s)]  # 右下
    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]  # 右上
    ordered[3] = corners[np.argmax(diff)]  # 左下

    # 计算目标尺寸
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    max_height = int(max(height_left, height_right))

    # 标准化到 140x40 (中国蓝牌标准比例)
    target_w, target_h = 140, 40
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    plate = cv2.warpPerspective(image, matrix, (target_w, target_h))

    return plate


def segment_and_save_chars(plate_image, plate_text, segmenter, output_dir, idx):
    """
    分割车牌字符并保存
    每个字符保存为: output_dir/<char>/<idx>.png
    """
    # 分割字符
    result = segmenter.segment(plate_image, method="projection")

    chars = result["chars"]
    num_expected = len(plate_text)

    if len(chars) != num_expected:
        return 0  # 分割数量不匹配，跳过

    saved = 0
    for i, (char_img, true_char) in enumerate(zip(chars, plate_text)):
        char_dir = output_dir / true_char
        char_dir.mkdir(parents=True, exist_ok=True)

        save_path = char_dir / f"{idx}_{i}.png"
        cv2.imwrite(str(save_path), char_img)
        saved += 1

    return saved


def prepare_ccpd(ccpd_dir, output_dir, max_samples=None, split_ratio=(0.7, 0.15, 0.15)):
    """
    主函数: 准备 CCPD 数据集

    参数:
        ccpd_dir: CCPD 数据集根目录
        output_dir: 输出目录
        max_samples: 最大处理样本数 (None=全部)
        split_ratio: (train, val, test) 比例
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)

    # 子集目录
    subsets = ["ccpd_base", "ccpd_blur", "ccpd_challenge",
               "ccpd_db", "ccpd_fn", "ccpd_green",
               "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]

    # 收集所有图片路径
    all_images = []
    for subset in subsets:
        subset_dir = ccpd_path / subset
        if subset_dir.exists():
            images = list(subset_dir.glob("*.jpg"))
            all_images.extend(images)
            print(f"  {subset}: {len(images)} 张")

    if not all_images:
        print("错误: 未找到 CCPD 图片，请检查路径")
        print(f"  搜索路径: {ccpd_path}")
        return

    print(f"总计: {len(all_images)} 张图片")

    # 限制数量
    if max_samples and len(all_images) > max_samples:
        random.seed(42)
        all_images = random.sample(all_images, max_samples)
        print(f"采样: {len(all_images)} 张")

    # 创建输出目录
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    plates_dir = output_path / "plates"  # 保存裁剪后的车牌图

    for d in [train_dir, val_dir, test_dir, plates_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 初始化分割器
    segmenter = CharSegmenter()

    # 统计
    stats = defaultdict(int)
    total_saved = 0
    total_plates = 0
    failed = 0

    # 随机划分
    random.seed(42)
    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    # n_test = n_total - n_train - n_val

    for i, img_path in enumerate(all_images):
        if i % 500 == 0:
            print(f"  进度: {i}/{n_total} (已保存 {total_saved} 个字符)")

        # 解析文件名
        annotation = parse_ccpd_filename(img_path.name)
        if annotation is None:
            failed += 1
            continue

        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
            continue

        # 裁剪车牌
        plate = crop_plate_from_ccpd(image, annotation)
        if plate is None or plate.size == 0:
            failed += 1
            continue

        plate_text = annotation["plate_text"]

        # 保存裁剪的车牌图
        plate_save_path = plates_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(plate_save_path), plate)
        total_plates += 1

        # 确定输出目录
        if i < n_train:
            char_output_dir = train_dir
        elif i < n_train + n_val:
            char_output_dir = val_dir
        else:
            char_output_dir = test_dir

        # 分割并保存字符
        saved = segment_and_save_chars(
            plate, plate_text, segmenter, char_output_dir, i
        )
        total_saved += saved
        stats[plate_text[0]] += 1  # 统计省份分布

    # 打印统计信息
    print("\n" + "=" * 50)
    print("数据集准备完成!")
    print(f"  总图片: {n_total}")
    print(f"  成功裁剪车牌: {total_plates}")
    print(f"  成功分割字符: {total_saved}")
    print(f"  失败: {failed}")
    print(f"\n  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  测试集: {test_dir}")
    print(f"  车牌图: {plates_dir}")

    # 打印字符分布
    print("\n省份分布 (前10):")
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_stats[:10]:
        print(f"  {char}: {count}")

    # 保存标签映射
    label_map = {}
    idx = 0
    for p in CHAR_RECOGNITION_CONFIG["provinces"]:
        label_map[p] = idx
        idx += 1
    for l in CHAR_RECOGNITION_CONFIG["letters"]:
        label_map[l] = idx
        idx += 1
    for d in CHAR_RECOGNITION_CONFIG["digits"]:
        label_map[d] = idx
        idx += 1

    import json
    with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"\n标签映射: {output_path / 'label_map.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCPD 数据集准备工具")
    parser.add_argument("--ccpd_dir", type=str, required=True,
                        help="CCPD 数据集根目录")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大处理样本数")
    parser.add_argument("--split", type=float, nargs=3,
                        default=(0.7, 0.15, 0.15),
                        help="训练/验证/测试 比例")

    args = parser.parse_args()

    print("=" * 50)
    print("CCPD 数据集准备工具")
    print(f"  数据集路径: {args.ccpd_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  最大样本数: {args.max_samples or '全部'}")
    print(f"  划分比例: {args.split}")
    print("=" * 50)

    prepare_ccpd(
        ccpd_dir=args.ccpd_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split_ratio=tuple(args.split),
    )
