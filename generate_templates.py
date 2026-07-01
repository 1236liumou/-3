"""
字符模板生成脚本

功能:
  1. 从 CCPD 数据集中提取字符模板 (推荐)
  2. 使用 OpenCV 渲染生成合成模板 (备选，无需数据集)

模板规格:
  - 尺寸: 20x28 像素
  - 格式: 二值化 PNG (黑底白字)
  - 目录: data/templates/{provinces,letters,digits}/<char>.png

使用方式:
    # 从 CCPD 数据集生成 (推荐)
    python scripts/generate_templates.py --ccpd_dir /path/to/CCPD --method ccpd

    # 生成合成模板 (无需数据集)
    python scripts/generate_templates.py --method synthetic
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.config import CHAR_RECOGNITION_CONFIG, TEMPLATE_DIR, DATA_DIR
from lpr.utils import parse_ccpd_filename, imwrite_unicode, imread_unicode
from lpr.char_segmentation import CharSegmenter


def generate_synthetic_templates(output_dir):
    """
    生成合成模板 (使用 OpenCV 文字渲染)
    注意: 数字和字母效果尚可，汉字需要系统中文字体支持
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size = CHAR_RECOGNITION_CONFIG["cnn_input_size"]  # (20, 28)
    w, h = size

    # ---- 数字模板 ----
    digits_dir = output_dir / "digits"
    digits_dir.mkdir(exist_ok=True)
    for d in CHAR_RECOGNITION_CONFIG["digits"]:
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(canvas, d, (3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 255, 2, cv2.LINE_AA)
        imwrite_unicode(str(digits_dir / f"{d}.png"), canvas)

    # ---- 字母模板 ----
    letters_dir = output_dir / "letters"
    letters_dir.mkdir(exist_ok=True)
    for l in CHAR_RECOGNITION_CONFIG["letters"]:
        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(canvas, l, (3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 255, 2, cv2.LINE_AA)
        imwrite_unicode(str(letters_dir / f"{l}.png"), canvas)

    # ---- 省份汉字模板 (使用系统中文字体) ----
    provinces_dir = output_dir / "provinces"
    provinces_dir.mkdir(exist_ok=True)

    # 尝试加载系统中文字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",       # Windows 黑体
        "C:/Windows/Fonts/msyh.ttc",         # Windows 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",       # Windows 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿
        "/System/Library/Fonts/PingFang.ttc",             # macOS 苹方
    ]

    font = None
    from PIL import ImageFont, ImageDraw, Image

    for fp in font_paths:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 22)
                print(f"  使用字体: {fp}")
                break
            except Exception:
                continue

    if font is None:
        print("  警告: 未找到中文字体，省份汉字模板将使用拼音首字母代替")
        # 退回到拼音方案
        province_pinyin = {
            "京": "B", "沪": "S", "津": "T", "渝": "C", "冀": "H",
            "晋": "J", "蒙": "M", "辽": "L", "吉": "J", "黑": "H",
            "苏": "S", "浙": "Z", "皖": "W", "闽": "F", "赣": "G",
            "鲁": "S", "豫": "Y", "鄂": "E", "湘": "X", "粤": "G",
            "桂": "G", "琼": "Q", "川": "S", "贵": "G", "云": "Y",
            "藏": "Z", "陕": "S", "甘": "G", "青": "Q", "宁": "N", "新": "X",
        }
        for p, py in province_pinyin.items():
            canvas = np.zeros((h, w), dtype=np.uint8)
            cv2.putText(canvas, py, (3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, 255, 2, cv2.LINE_AA)
            imwrite_unicode(str(provinces_dir / f"{p}.png"), canvas)
    else:
        for p in CHAR_RECOGNITION_CONFIG["provinces"]:
            pil_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(pil_img)
            # 居中绘制
            bbox = draw.textbbox((0, 0), p, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (w - text_w) // 2 - bbox[0]
            y = (h - text_h) // 2 - bbox[1]
            draw.text((x, y), p, fill=255, font=font)
            canvas = np.array(pil_img)
            imwrite_unicode(str(provinces_dir / f"{p}.png"), canvas)

    print(f"合成模板已保存到: {output_dir}")
    print(f"  数字: {len(CHAR_RECOGNITION_CONFIG['digits'])} 个")
    print(f"  字母: {len(CHAR_RECOGNITION_CONFIG['letters'])} 个")
    print(f"  省份: {len(CHAR_RECOGNITION_CONFIG['provinces'])} 个")


def generate_templates_from_ccpd(ccpd_dir, output_dir, max_per_char=20):
    """
    从 CCPD 数据集提取字符模板
    对每种字符收集多个样本，取质量最好的作为模板

    参数:
        ccpd_dir: CCPD 数据集目录
        output_dir: 输出目录
        max_per_char: 每种字符最多收集多少个样本
    """
    ccpd_path = Path(ccpd_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    subsets = ["ccpd_base", "ccpd_blur", "ccpd_challenge",
               "ccpd_db", "ccpd_fn", "ccpd_green",
               "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]

    all_images = []
    for subset in subsets:
        subset_dir = ccpd_path / subset
        if subset_dir.exists():
            all_images.extend(list(subset_dir.glob("*.jpg")))

    if not all_images:
        print(f"错误: 未在 {ccpd_path} 找到 CCPD 图片")
        return

    print(f"找到 {len(all_images)} 张 CCPD 图片")

    segmenter = CharSegmenter()

    # 按字符收集样本
    char_samples = {}  # {char: [(img, quality_score), ...]}

    processed = 0
    for img_path in all_images:
        if processed % 1000 == 0:
            print(f"  进度: {processed}/{len(all_images)}")

        annotation = parse_ccpd_filename(img_path.name)
        if annotation is None:
            continue

        image = imread_unicode(str(img_path))
        if image is None:
            continue

        # 使用标注的四角点裁剪车牌
        corners = np.array(annotation["corners"], dtype="float32")
        s = corners.sum(axis=1)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = corners[np.argmin(s)]
        ordered[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        ordered[1] = corners[np.argmin(diff)]
        ordered[3] = corners[np.argmax(diff)]

        target_w, target_h = 140, 40
        dst = np.array([
            [0, 0], [target_w - 1, 0],
            [target_w - 1, target_h - 1], [0, target_h - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered, dst)
        plate = cv2.warpPerspective(image, matrix, (target_w, target_h))

        # 分割字符
        result = segmenter.segment(plate, method="projection")
        chars = result["chars"]
        plate_text = annotation["plate_text"]

        if len(chars) != len(plate_text):
            continue

        # 收集每个字符样本
        for char_img, true_char in zip(chars, plate_text):
            if true_char not in char_samples:
                char_samples[true_char] = []
            # 简单质量评分: 字符区域的像素密度
            density = np.sum(char_img > 0) / char_img.size
            if 0.1 < density < 0.8:  # 合理的字符密度范围
                char_samples[true_char].append((char_img, density))

        processed += 1

        # 限制总处理量
        if processed >= 50000:
            break

    # 为每个字符选择最佳模板
    print(f"\n收集到 {len(char_samples)} 种字符")

    for category, chars in [
        ("provinces", CHAR_RECOGNITION_CONFIG["provinces"]),
        ("letters", CHAR_RECOGNITION_CONFIG["letters"]),
        ("digits", CHAR_RECOGNITION_CONFIG["digits"]),
    ]:
        cat_dir = output_path / category
        cat_dir.mkdir(exist_ok=True)

        for char in chars:
            samples = char_samples.get(char, [])
            if not samples:
                print(f"  警告: 字符 '{char}' 无样本")
                continue

            # 选择密度最接近 0.4 的样本 (适中)
            samples.sort(key=lambda x: abs(x[1] - 0.4))
            best_img = samples[0][0]

            imwrite_unicode(str(cat_dir / f"{char}.png"), best_img)

    # 统计
    total = sum(len(v) for v in char_samples.values())
    print(f"\n模板生成完成!")
    print(f"  字符种类: {len(char_samples)}")
    print(f"  总样本数: {total}")
    print(f"  输出目录: {output_path}")

    # 打印每种字符的样本数
    print("\n各字符样本数:")
    for char in CHAR_RECOGNITION_CONFIG["provinces"]:
        count = len(char_samples.get(char, []))
        print(f"  {char}: {count}", end="  ")
        if (list(CHAR_RECOGNITION_CONFIG["provinces"]).index(char) + 1) % 8 == 0:
            print()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="字符模板生成工具")
    parser.add_argument("--method", type=str, default="synthetic",
                        choices=["synthetic", "ccpd"],
                        help="生成方法: synthetic (合成) 或 ccpd (从CCPD提取)")
    parser.add_argument("--ccpd_dir", type=str, default=None,
                        help="CCPD 数据集路径 (method=ccpd 时需要)")
    parser.add_argument("--output_dir", type=str,
                        default=str(TEMPLATE_DIR),
                        help="输出目录")
    parser.add_argument("--max_per_char", type=int, default=20,
                        help="每种字符最大样本数 (ccpd方法)")

    args = parser.parse_args()

    print("=" * 50)
    print("字符模板生成工具")
    print(f"  方法: {args.method}")
    print(f"  输出: {args.output_dir}")
    print("=" * 50)

    if args.method == "ccpd":
        if not args.ccpd_dir:
            print("错误: ccpd 方法需要 --ccpd_dir 参数")
            sys.exit(1)
        generate_templates_from_ccpd(
            ccpd_dir=args.ccpd_dir,
            output_dir=args.output_dir,
            max_per_char=args.max_per_char,
        )
    else:
        generate_synthetic_templates(args.output_dir)
