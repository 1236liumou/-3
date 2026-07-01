# 车牌识别系统

> 数字图像处理课程设计项目

## 项目简介

本项目实现了一个端到端的车牌识别系统，包含图像预处理、车牌定位、倾斜校正、字符分割、字符识别五个核心阶段，并配备 Streamlit 交互式可视化界面。

## 系统架构

```
输入车辆图像
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Stage 1    │────▶│  Stage 2    │────▶│  Stage 3    │────▶│  Stage 4    │────▶│  Stage 5    │
│  图像预处理  │     │  车牌定位    │     │  倾斜校正    │     │  字符分割    │     │  字符识别    │
│             │     │             │     │             │     │             │     │             │
│ ·灰度化      │     │ ·HSV颜色过滤 │     │ ·Hough直线   │     │ ·垂直投影    │     │ ·模板匹配    │
│ ·双边滤波    │     │ ·形态学闭操作│     │  检测        │     │  分割       │     │ ·SVM        │
│ ·CLAHE增强   │     │ ·轮廓检测    │     │ ·仿射变换    │     │ ·连通域分析  │     │ ·CNN        │
│             │     │ ·候选筛选    │     │  旋转校正    │     │ ·字符标准化  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
                                                            输出车牌字符串
                                                           如: 京A12345
```

## 目录结构

```
车牌识别系统/
├── lpr/                        # 核心算法包
│   ├── __init__.py
│   ├── config.py               # 全局配置 (所有可调参数)
│   ├── utils.py                # 工具函数 (计时、绘图、校验等)
│   ├── preprocess.py           # Stage 1: 图像预处理
│   ├── plate_locator.py        # Stage 2: 车牌定位
│   ├── skew_correction.py      # Stage 3: 倾斜校正
│   ├── char_segmentation.py    # Stage 4: 字符分割
│   ├── char_recognition.py     # Stage 5: 字符识别
│   └── pipeline.py             # 端到端流水线
├── app.py                      # Streamlit 可视化界面
├── requirements.txt            # Python 依赖
└── README.md                   # 项目文档
```

## 环境配置

```bash
# 1. 创建虚拟环境 (推荐 Python 3.10+)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 命令行使用

```python
from lpr.pipeline import LPRPipeline

# 创建流水线
pipeline = LPRPipeline(
    locate_method="color",        # 车牌定位方法
    segment_method="projection",  # 字符分割方法
    recognize_method="template",  # 字符识别方法
    verbose=True,
)

# 识别单张图片
result = pipeline.recognize("path/to/car.jpg")

print(f"车牌号: {result['plate_text']}")
print(f"置信度: {result['confidence']:.1%}")
print(f"总耗时: {result['total_time']:.0f}ms")

# 批量识别
results = pipeline.recognize_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### 启动可视化界面

```bash
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 各阶段算法详解

### Stage 1: 图像预处理

| 步骤 | 算法 | 作用 |
|------|------|------|
| 灰度化 | `cv2.cvtColor(BGR2GRAY)` | 彩色转灰度，减少计算量 |
| 双边滤波 | `cv2.bilateralFilter` | 保边去噪，保留车牌边缘信息 |
| CLAHE | `cv2.createCLAHE` | 自适应直方图均衡化，增强对比度 |

**核心知识点**: 空域滤波、直方图处理、边缘保持滤波

### Stage 2: 车牌定位

| 步骤 | 算法 | 作用 |
|------|------|------|
| 颜色过滤 | HSV颜色空间 `cv2.inRange` | 提取蓝/绿/黄车牌区域 |
| 形态学操作 | `cv2.morphologyEx(CLOSE)` | 连接字符区域，形成连通块 |
| 轮廓检测 | `cv2.findContours` | 检测候选区域 |
| 区域筛选 | 面积+宽高比+矩形度 | 过滤非车牌区域 |

**核心知识点**: 颜色空间转换、形态学运算、轮廓分析

### Stage 3: 倾斜校正

| 步骤 | 算法 | 作用 |
|------|------|------|
| 边缘检测 | `cv2.Canny` | 检测车牌边缘 |
| 直线检测 | `cv2.HoughLinesP` | 检测车牌上下边框线 |
| 角度计算 | 统计直线角度中位数 | 估计倾斜角度 |
| 旋转变换 | `cv2.warpAffine` | 仿射变换旋转校正 |

**核心知识点**: Hough变换、仿射变换、几何校正

### Stage 4: 字符分割

| 步骤 | 算法 | 作用 |
|------|------|------|
| 标准化 | `cv2.resize` | 统一车牌尺寸 |
| 二值化 | Otsu `cv2.threshold` | 字符与背景分离 |
| 去噪 | 连通域分析 | 去除铆钉等小噪声 |
| 投影分割 | 垂直投影波谷检测 | 按列投影找字符边界 |
| 标准化 | 等比缩放+居中填充 | 统一字符尺寸 |

**核心知识点**: 图像二值化、投影分析、连通域标记

### Stage 5: 字符识别

提供三种方案:

| 方法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| 模板匹配 | `cv2.matchTemplate` 归一化互相关 | 无需训练，开箱即用 | 准确率有限，对字体敏感 |
| SVM | HOG特征 + RBF核SVM | 训练数据需求少，泛化好 | 特征工程依赖经验 |
| CNN | 卷积神经网络端到端学习 | 准确率最高，鲁棒性强 | 需要大量训练数据 |

**核心知识点**: 模板匹配、特征提取(HOG)、机器学习分类器、深度学习

## 评估指标

| 指标 | 说明 |
|------|------|
| 定位准确率 | 正确定位车牌的图像数 / 总图像数 |
| 字符级准确率 | 正确识别的字符数 / 总字符数 |
| 端到端准确率 | 车牌完全识别正确的图像数 / 总图像数 |
| 平均处理时间 | 单张图片从输入到输出的毫秒数 |
| FPS | 每秒处理帧数 (1000 / 平均处理时间) |

## 扩展功能

- [x] 多车牌类型支持 (蓝牌/绿牌/黄牌)
- [x] 倾斜车牌校正
- [x] 车牌格式校验
- [x] Streamlit 可视化界面
- [x] 各阶段中间结果展示
- [x] 性能耗时分析
- [ ] 视频流实时识别
- [ ] 多车牌同时检测
- [ ] SVM/CNN 模型训练脚本
- [ ] 结果数据库存储

## 数据集

推荐使用 CCPD (Chinese City Parking Dataset):
- 仓库: https://github.com/detectRecog/CCPD
- 包含 25 万+ 张中国车牌图像
- 标注信息: 车牌四角坐标 + 车牌号

## 参考文献

1. Gonzalez & Woods, *Digital Image Processing*, 4th Edition
2. OpenCV Documentation: https://docs.opencv.org
3. Li, H., & Wang, P. (2018). CCPD: Chinese City Parking Dataset
