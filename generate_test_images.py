"""
合成测试图片生成脚本

生成包含车牌的合成车辆图片，用于系统测试和演示
无需真实数据集，适合方案一（开箱即用）

生成的图片包含:
  1. 模拟车辆背景 (渐变色 + 噪声)
  2. 蓝底白字车牌 (标准中国蓝牌格式)
  3. 可选绿牌 (新能源)

使用方式:
    python scripts/generate_test_images.py
    python scripts/generate_test_images.py --num 10 --output_dir data/test_images
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.config import CHAR_RECOGNITION_CONFIG, TEST_IMAGES_DIR
from lpr.utils import imwrite_unicode

# 车牌字符集
PROVINCES = CHAR_RECOGNITION_CONFIG["provinces"]
LETTERS = CHAR_RECOGNITION_CONFIG["letters"]
DIGITS = CHAR_RECOGNITION_CONFIG["digits"]


def generate_plate_number(plate_type="blue"):
    """随机生成车牌号"""
    province = random.choice(PROVINCES)
    letter = random.choice(LETTERS)

    if plate_type == "green":
        # 新能源: 省份+字母+D/F+5位  如: 京AD12345
        second = random.choice(["D", "F"])
        rest = "".join(random.choice(DIGITS + LETTERS) for _ in range(5))
        return f"{province}{letter}{second}{rest}"
    else:
        # 蓝牌: 省份+字母+5位  如: 京A12345
        rest = "".join(random.choice(DIGITS + LETTERS) for _ in range(5))
        return f"{province}{letter}{rest}"


def render_plate_image(plate_text, plate_type="blue"):
    """
    渲染车牌图像 (蓝底白字 / 绿底白字)
    返回: 车牌图像 (BGR)
    """
    # 标准车牌比例 440:140 ≈ 3.14:1
    plate_w, plate_h = 280, 90

    # 车牌底色
    if plate_type == "green":
        bg_color = (140, 165, 50)  # 绿牌 (BGR: 黄绿)
    else:
        bg_color = (180, 40, 40)   # 蓝牌 (BGR: 蓝色)

    plate = np.full((plate_h, plate_w, 3), bg_color, dtype=np.uint8)

    # 添加白色边框
    cv2.rectangle(plate, (2, 2), (plate_w - 3, plate_h - 3),
                  (220, 220, 220), 1)

    # 使用 PIL 渲染中文+英文+数字
    from PIL import ImageFont, ImageDraw, Image

    # 加载字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 52)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    pil_plate = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_plate)

    # 逐字符绘制，模拟真实车牌排版
    # 第一个汉字 (省份) 位置稍偏左
    char_spacing = 33  # 字符间距
    start_x = 12

    for i, char in enumerate(plate_text):
        if i == 0:
            # 省份汉字
            x = start_x
        elif i == 1:
            # 字母 (和省份之间有间隔)
            x = start_x + 40
        else:
            x = start_x + 40 + (i - 1) * char_spacing

        # 居中绘制
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y = (plate_h - th) // 2 - bbox[1]
        draw.text((x, y), char, fill=(255, 255, 255), font=font)

    plate = cv2.cvtColor(np.array(pil_plate), cv2.COLOR_RGB2BGR)

    # 添加少量噪声使更真实
    noise = np.random.normal(0, 3, plate.shape).astype(np.int16)
    plate = np.clip(plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return plate


def generate_car_background(width=640, height=480):
    """
    生成模拟车辆背景
    使用渐变色 + 随机噪声模拟车身
    """
    # 随机选择车身颜色 (深色系，使蓝牌更突出)
    car_colors = [
        ((40, 40, 40), (60, 60, 60)),      # 黑色
        ((30, 30, 80), (50, 50, 120)),     # 深蓝
        ((60, 40, 30), (90, 60, 50)),      # 深棕
        ((30, 60, 30), (50, 90, 50)),      # 深绿
        ((80, 80, 80), (110, 110, 110)),   # 灰色
        ((120, 120, 60), (150, 150, 80)),  # 卡其
    ]
    color_start, color_end = random.choice(car_colors)

    # 垂直渐变
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        r = int(color_start[0] + (color_end[0] - color_start[0]) * ratio)
        g = int(color_start[1] + (color_end[1] - color_start[1]) * ratio)
        b = int(color_start[2] + (color_end[2] - color_start[2]) * ratio)
        bg[y, :] = [b, g, r]

    # 添加纹理噪声
    noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 模拟车灯 (两个亮区域)
    for _ in range(random.randint(2, 4)):
        lx = random.randint(width // 4, 3 * width // 4)
        ly = random.randint(height // 4, 3 * height // 4)
        lr = random.randint(20, 40)
        cv2.circle(bg, (lx, ly), lr, (200, 200, 200), -1)
        # 模糊
        bg = cv2.GaussianBlur(bg, (5, 5), 0)

    return bg


def compose_test_image(plate_text, plate_type="blue"):
    """
    合成一张测试图片: 车辆背景 + 车牌
    返回: (完整图像, 车牌在图中的位置 bbox)
    """
    width, height = 640, 480
    bg = generate_car_background(width, height)

    # 渲染车牌
    plate = render_plate_image(plate_text, plate_type)
    ph, pw = plate.shape[:2]

    # 随机放置车牌位置 (中下部区域)
    max_x = width - pw - 40
    max_y = height - ph - 40
    min_x = 40
    min_y = height // 3

    px = random.randint(min_x, max_x)
    py = random.randint(min_y, max_y)

    # 可选: 轻微旋转车牌 (模拟倾斜)
    angle = random.uniform(-3, 3)
    center = (pw // 2, ph // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    plate_rotated = cv2.warpAffine(plate, M, (pw, ph),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(180, 40, 40))

    # 将车牌贴到背景上
    bg[py:py + ph, px:px + pw] = plate_rotated

    # 轻微模糊整体图像 (模拟拍摄)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)

    # 车牌位置 (x, y, w, h)
    bbox = (px, py, pw, ph)

    return bg, bbox


def main():
    parser = argparse.ArgumentParser(description="生成合成测试图片")
    parser.add_argument("--num", type=int, default=8,
                        help="生成图片数量")
    parser.add_argument("--output_dir", type=str,
                        default=str(TEST_IMAGES_DIR),
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (可复现)")

    args = parser.parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("合成测试图片生成工具")
    print(f"  数量: {args.num}")
    print(f"  输出: {output_dir}")
    print("=" * 50)

    results = []

    for i in range(args.num):
        # 随机选择蓝牌或绿牌
        plate_type = random.choice(["blue", "blue", "blue", "green"])
        plate_text = generate_plate_number(plate_type)

        image, bbox = compose_test_image(plate_text, plate_type)

        # 保存图片
        img_path = output_dir / f"test_{i+1:03d}.jpg"
        imwrite_unicode(str(img_path), image)

        # 保存标注信息
        results.append({
            "image": str(img_path),
            "plate_text": plate_text,
            "plate_type": plate_type,
            "bbox": bbox,
        })

        print(f"  [{i+1}/{args.num}] {plate_text} ({plate_type}) -> {img_path.name}")

    # 保存标注文件
    import json
    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成! 共 {args.num} 张图片")
    print(f"标注文件: {ann_path}")
    print(f"\n车牌号列表:")
    for r in results:
        print(f"  {r['image'].split('/')[-1]}: {r['plate_text']} ({r['plate_type']})")


if __name__ == "__main__":
    main()
