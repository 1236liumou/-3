"""
模型评估脚本

功能:
  1. 使用 CCPD 测试集评估端到端识别准确率
  2. 分阶段评估 (定位准确率、分割准确率、识别准确率)
  3. 生成混淆矩阵和分类报告
  4. 输出性能分析报告

使用方式:
    python scripts/evaluate.py --ccpd_dir /path/to/CCPD --method template
    python scripts/evaluate.py --test_dir data/test/ --method svm
"""

import os
import cv2
import json
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lpr.pipeline import LPRPipeline
from lpr.utils import parse_ccpd_filename
from lpr.config import CHAR_RECOGNITION_CONFIG


def evaluate_end_to_end(pipeline, image_paths, verbose=True):
    """
    端到端评估

    返回:
        metrics: dict {
            "total": 总数,
            "success": 成功识别数,
            "correct": 完全正确数,
            "char_level_correct": 字符级正确数,
            "char_level_total": 字符级总数,
            "avg_time": 平均耗时,
            "details": 详细结果列表,
        }
    """
    total = len(image_paths)
    success = 0
    correct = 0
    char_correct = 0
    char_total = 0
    total_time = 0
    details = []

    for i, img_path in enumerate(image_paths):
        if verbose and (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total} "
                  f"(成功: {success}, 正确: {correct})")

        # 获取真实标签
        annotation = parse_ccpd_filename(Path(img_path).name)
        true_plate = annotation["plate_text"] if annotation else ""

        # 识别
        result = pipeline.recognize(str(img_path))
        total_time += result["total_time"]

        pred_plate = result["plate_text"]

        entry = {
            "filename": Path(img_path).name,
            "true_plate": true_plate,
            "pred_plate": pred_plate,
            "success": result["success"],
            "correct": pred_plate == true_plate,
            "confidence": result["confidence"],
            "time_ms": result["total_time"],
        }
        details.append(entry)

        if result["success"]:
            success += 1

            # 车牌级准确率
            if pred_plate == true_plate:
                correct += 1

            # 字符级准确率
            min_len = min(len(pred_plate), len(true_plate))
            for j in range(min_len):
                if pred_plate[j] == true_plate[j]:
                    char_correct += 1
            char_total += len(true_plate)

    metrics = {
        "total": total,
        "success": success,
        "correct": correct,
        "char_level_correct": char_correct,
        "char_level_total": char_total,
        "plate_accuracy": correct / total if total > 0 else 0,
        "plate_accuracy_of_success": correct / success if success > 0 else 0,
        "char_accuracy": char_correct / char_total if char_total > 0 else 0,
        "success_rate": success / total if total > 0 else 0,
        "avg_time": total_time / total if total > 0 else 0,
        "total_time": total_time,
        "details": details,
    }

    return metrics


def evaluate_by_stage(pipeline, image_paths, max_samples=500):
    """
    分阶段评估: 定位 -> 分割 -> 识别
    分析每个阶段对最终准确率的贡献
    """
    from lpr.char_segmentation import CharSegmenter

    segmenter = CharSegmenter()
    total = min(len(image_paths), max_samples)

    stage_stats = {
        "locate": {"success": 0, "total": 0},
        "segment": {"success": 0, "total": 0},
        "recognize": {"success": 0, "total": 0},
    }

    for i in range(total):
        img_path = image_paths[i]
        annotation = parse_ccpd_filename(Path(img_path).name)
        if not annotation:
            continue

        true_plate = annotation["plate_text"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Stage 1+2: 预处理 + 定位
        preprocess_result = pipeline.preprocessor.process(image)
        locate_result = pipeline.locator.locate(
            preprocess_result["color"],
            preprocess_result["gray"],
            method=pipeline.locate_method,
        )
        plate_region = locate_result["plate_region"]

        stage_stats["locate"]["total"] += 1
        if plate_region is not None:
            # 检查定位是否大致正确 (IoU)
            x, y, w, h = plate_region["rect"]
            true_bbox = annotation["bbox"]
            # 简单检查: 定位框是否覆盖了真实车牌区域
            tx1, ty1, tx2, ty2 = true_bbox
            if x < tx2 and x + w > tx1 and y < ty2 and y + h > ty1:
                stage_stats["locate"]["success"] += 1

                # 截取车牌
                pad = max(w, h) // 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                plate_image = image[y1:y2, x1:x2]

                # Stage 3: 校正
                corrected, _, _ = pipeline.corrector.correct(
                    plate_image, plate_region, method="hough"
                )

                # Stage 4: 分割
                seg_result = segmenter.segment(corrected, method="projection")
                stage_stats["segment"]["total"] += 1
                if seg_result["num_chars"] == len(true_plate):
                    stage_stats["segment"]["success"] += 1

                    # Stage 5: 识别
                    plate_text, _, avg_conf, _ = pipeline.recognizer.recognize_all(
                        seg_result["chars"]
                    )
                    stage_stats["recognize"]["total"] += 1
                    if plate_text == true_plate:
                        stage_stats["recognize"]["success"] += 1

    # 计算各阶段准确率
    result = {}
    for stage, stats in stage_stats.items():
        if stats["total"] > 0:
            result[stage] = {
                "success": stats["success"],
                "total": stats["total"],
                "accuracy": stats["success"] / stats["total"],
            }
        else:
            result[stage] = {"success": 0, "total": 0, "accuracy": 0}

    return result


def generate_report(metrics, stage_metrics, output_path):
    """生成评估报告"""
    report = []
    report.append("=" * 60)
    report.append("车牌识别系统评估报告")
    report.append("=" * 60)
    report.append("")

    report.append("一、端到端评估")
    report.append("-" * 40)
    report.append(f"  测试样本总数:  {metrics['total']}")
    report.append(f"  成功识别数:    {metrics['success']}")
    report.append(f"  识别成功率:    {metrics['success_rate']:.2%}")
    report.append(f"  车牌完全正确:  {metrics['correct']}")
    report.append(f"  车牌级准确率:  {metrics['plate_accuracy']:.2%}")
    report.append(f"  (占成功识别):  {metrics['plate_accuracy_of_success']:.2%}")
    report.append(f"  字符级准确率:  {metrics['char_accuracy']:.2%}")
    report.append(f"  平均耗时:      {metrics['avg_time']:.1f} ms")
    report.append(f"  总耗时:        {metrics['total_time']:.0f} ms")
    report.append("")

    if stage_metrics:
        report.append("二、分阶段评估")
        report.append("-" * 40)
        stage_names = {
            "locate": "车牌定位",
            "segment": "字符分割",
            "recognize": "字符识别",
        }
        for stage_key in ["locate", "segment", "recognize"]:
            s = stage_metrics[stage_key]
            name = stage_names[stage_key]
            report.append(f"  {name}:")
            report.append(f"    样本数:  {s['total']}")
            report.append(f"    成功数:  {s['success']}")
            report.append(f"    准确率:  {s['accuracy']:.2%}")
        report.append("")

    report.append("三、错误分析")
    report.append("-" * 40)
    # 分析失败原因
    failed = [d for d in metrics["details"] if not d["success"]]
    wrong = [d for d in metrics["details"] if d["success"] and not d["correct"]]

    report.append(f"  识别失败 (未检测到车牌): {len(failed)}")
    report.append(f"  识别错误 (检测到但内容不对): {len(wrong)}")

    if wrong:
        # 分析字符级错误分布
        error_chars = defaultdict(int)
        for d in wrong:
            pred = d["pred_plate"]
            true = d["true_plate"]
            min_len = min(len(pred), len(true))
            for j in range(min_len):
                if pred[j] != true[j]:
                    error_chars[f"{true[j]}->{pred[j]}"] += 1

        report.append("\n  常见字符错误 (前10):")
        sorted_errors = sorted(error_chars.items(), key=lambda x: x[1], reverse=True)
        for error, count in sorted_errors[:10]:
            report.append(f"    {error}: {count} 次")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    # 保存报告
    report_path = Path(output_path) / "evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")

    # 保存详细结果 JSON
    json_path = Path(output_path) / "evaluation_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {k: v for k, v in metrics.items() if k != "details"},
            "stage_metrics": stage_metrics,
            "details": metrics["details"],
        }, f, ensure_ascii=False, indent=2)
    print(f"详细结果: {json_path}")


def evaluate(ccpd_dir=None, test_dir=None, method="template",
             max_samples=None, output_dir="output/evaluation"):
    """
    主评估函数

    参数:
        ccpd_dir: CCPD 数据集目录 (优先使用)
        test_dir: 测试数据目录 (备选)
        method: 识别方法
        max_samples: 最大评估样本数
        output_dir: 输出目录
    """
    # 收集测试图片
    image_paths = []

    if ccpd_dir:
        ccpd_path = Path(ccpd_dir)
        # 使用 base 或 fn 子集做测试
        for subset in ["ccpd_base", "ccpd_fn"]:
            subset_dir = ccpd_path / subset
            if subset_dir.exists():
                image_paths.extend(list(subset_dir.glob("*.jpg")))

    if test_dir:
        test_path = Path(test_dir)
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            image_paths.extend(test_path.glob(ext))

    if not image_paths:
        print("错误: 未找到测试图片")
        print("请指定 --ccpd_dir 或 --test_dir")
        return

    # 限制样本数
    if max_samples and len(image_paths) > max_samples:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, max_samples)

    print(f"评估样本数: {len(image_paths)}")

    # 初始化流水线
    pipeline = LPRPipeline(
        locate_method="color",
        segment_method="projection",
        recognize_method=method,
        verbose=False,
    )

    # 端到端评估
    print("\n正在进行端到端评估...")
    metrics = evaluate_end_to_end(pipeline, image_paths)

    # 分阶段评估 (采样)
    print("\n正在进行分阶段评估...")
    stage_metrics = evaluate_by_stage(pipeline, image_paths, max_samples=min(500, len(image_paths)))

    # 生成报告
    generate_report(metrics, stage_metrics, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="车牌识别系统评估")
    parser.add_argument("--ccpd_dir", type=str, default=None,
                        help="CCPD 数据集目录")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="测试数据目录")
    parser.add_argument("--method", type=str, default="template",
                        choices=["template", "svm", "cnn"],
                        help="识别方法")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数")
    parser.add_argument("--output_dir", type=str,
                        default="output/evaluation",
                        help="输出目录")

    args = parser.parse_args()

    evaluate(
        ccpd_dir=args.ccpd_dir,
        test_dir=args.test_dir,
        method=args.method,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
