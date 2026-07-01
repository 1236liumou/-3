"""
批量测试脚本

功能:
  1. 对指定目录下的所有图片进行批量车牌识别
  2. 输出结果到 CSV 文件
  3. 生成可视化结果图

使用方式:
    python scripts/batch_test.py --input_dir test_images/ --output_dir output/batch/
"""

import os
import cv2
import csv
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.pipeline import LPRPipeline
from lpr.utils import resize_with_ratio


def batch_test(input_dir, output_dir,
               locate_method="color", segment_method="projection",
               recognize_method="template", save_results=True):
    """
    批量测试

    参数:
        input_dir: 输入图片目录
        output_dir: 输出目录
        locate_method: 定位方法
        segment_method: 分割方法
        recognize_method: 识别方法
        save_results: 是否保存结果图
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))

    if not image_files:
        print(f"错误: 在 {input_path} 中未找到图片")
        return

    print("=" * 60)
    print("批量车牌识别测试")
    print("=" * 60)
    print(f"  输入目录: {input_path}")
    print(f"  输出目录: {output_path}")
    print(f"  图片数量: {len(image_files)}")
    print(f"  定位方法: {locate_method}")
    print(f"  分割方法: {segment_method}")
    print(f"  识别方法: {recognize_method}")
    print("=" * 60)

    # 初始化流水线
    pipeline = LPRPipeline(
        locate_method=locate_method,
        segment_method=segment_method,
        recognize_method=recognize_method,
        verbose=False,
    )

    # 结果列表
    results = []
    success_count = 0
    total_time = 0

    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")

        result = pipeline.recognize(str(img_path))
        total_time += result["total_time"]

        entry = {
            "filename": img_path.name,
            "success": result["success"],
            "plate_text": result["plate_text"],
            "confidence": f"{result['confidence']:.4f}",
            "is_valid": result["is_valid"],
            "time_ms": f"{result['total_time']:.2f}",
            "error": result.get("error", ""),
        }

        if result["success"]:
            success_count += 1
            print(f"  车牌: {result['plate_text']}  "
                  f"置信度: {result['confidence']:.2%}  "
                  f"耗时: {result['total_time']:.0f}ms")

            # 保存结果图
            if save_results and result["result_image"] is not None:
                result_path = output_path / f"result_{img_path.name}"
                cv2.imwrite(str(result_path), result["result_image"])
        else:
            print(f"  失败: {result.get('error', '未知错误')}")

        results.append(entry)

    # ============================================================
    # 输出统计
    # ============================================================
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"  总图片数: {len(image_files)}")
    print(f"  成功识别: {success_count}")
    print(f"  失败: {len(image_files) - success_count}")
    print(f"  成功率: {success_count / len(image_files):.2%}")
    print(f"  总耗时: {total_time:.0f}ms")
    print(f"  平均耗时: {total_time / len(image_files):.1f}ms")
    if total_time > 0:
        print(f"  平均帧率: {1000 * len(image_files) / total_time:.1f} FPS")

    # ============================================================
    # 保存 CSV
    # ============================================================
    csv_path = output_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "success", "plate_text", "confidence",
            "is_valid", "time_ms", "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存: {csv_path}")

    # 保存成功识别的车牌列表
    success_path = output_path / "recognized_plates.txt"
    with open(success_path, "w", encoding="utf-8") as f:
        for r in results:
            if r["success"]:
                f.write(f"{r['filename']}\t{r['plate_text']}\t"
                        f"{r['confidence']}\n")

    print(f"识别列表: {success_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量车牌识别测试")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入图片目录")
    parser.add_argument("--output_dir", type=str, default="output/batch/",
                        help="输出目录")
    parser.add_argument("--locate_method", type=str, default="color",
                        choices=["color", "multi", "edge"])
    parser.add_argument("--segment_method", type=str, default="projection",
                        choices=["projection", "connected"])
    parser.add_argument("--recognize_method", type=str, default="template",
                        choices=["template", "svm", "cnn"])
    parser.add_argument("--no_save", action="store_true",
                        help="不保存结果图")

    args = parser.parse_args()

    batch_test(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        locate_method=args.locate_method,
        segment_method=args.segment_method,
        recognize_method=args.recognize_method,
        save_results=not args.no_save,
    )
