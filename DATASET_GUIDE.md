# 车牌识别系统 · 数据集获取指南

> 本文档详细介绍项目可用的数据集来源、下载方式、标注格式说明及自建数据集方法

---

## 一、CCPD 数据集（强烈推荐）

### 1.1 简介

CCPD（Chinese City Parking Dataset）是目前最大规模的中国车牌开源数据集，由中国科学技术大学团队创建，在 ECCV 2018 会议上发表。该项目使用 MIT 开源协议，非常适合学术研究和课程设计使用。

- **GitHub 仓库**: https://github.com/detectRecog/CCPD
- **总图像数**: 30 万+ 张
- **分辨率**: 720×1160 像素
- **标注方式**: 标注信息嵌入在文件名中（无需额外标注文件）

### 1.2 子集说明

CCPD 包含多个子集，覆盖不同难度场景：

| 子集名称 | 说明 | 图像数量 | 适用阶段 |
|----------|------|----------|----------|
| **CCPD-Base** | 基础场景，正常光照、正面拍摄 | ~20 万 | 训练集 + 验证集 |
| **CCPD-DB** | Dark / Bright，过暗或过亮场景 | ~2 万 | 测试集 |
| **CCPD-Blur** | 模糊车牌（运动模糊、失焦） | ~2 万 | 测试集 |
| **CCPD-FN** | Far / Near，远距离或近距离拍摄 | ~2 万 | 测试集 |
| **CCPD-Rotate** | 旋转车牌（水平倾斜） | ~1 万 | 测试集 |
| **CCPD-Tilt** | 倾斜车牌（透视变形） | ~3 万 | 测试集 |
| **CCPD-Challenge** | 综合挑战场景（最难） | ~5 万 | 测试集 |
| **CCPD-Green** | 新能源绿牌（8 位车牌号） | ~3 万 | 训练 + 测试 |

### 1.3 下载方式

**CCPD2019 主数据集**（含 Base / DB / Blur / FN / Rotate / Tilt / Challenge）:

- **Google Drive**: https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc
- **百度网盘**: https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw （提取码: hm0u）
- **文件大小**: 约 12 GB（.tar.xz 压缩格式）
- **解压命令**: `tar xf CCPD2019.tar.xz`

**CCPD-Green 新能源车牌子集**:

- **Google Drive**: https://drive.google.com/file/d/1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR/view
- **百度网盘**: https://pan.baidu.com/s/1JSpc9BZXFlPkXxRK4qUCyw （提取码: ol3j）

### 1.4 标注格式

CCPD 的标注信息直接编码在文件名中，无需额外的标注文件。

文件名格式示例:
```
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
```

按 `-` 分割为 7 个字段:

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 1. 面积比 | `025` | 车牌面积占整图面积的比例 × 100 |
| 2. 倾斜度 | `95_113` | 水平倾斜角_垂直倾斜角（度） |
| 3. 外接矩形 | `154&383_386&473` | 左上角_右下角坐标 (x&y 格式) |
| 4. 四角坐标 | `386&473_177&454_154&383_363&402` | 车牌四角精确坐标（右下→左下→左上→右上） |
| 5. 车牌号索引 | `0_0_22_27_27_33_16` | 每个字符在字符表中的索引 |
| 6. 亮度 | `37` | 车牌区域亮度值 |
| 7. 模糊度 | `15` | 车牌区域模糊度值 |

**字符索引对照表**:

```python
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
       'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
# 注意: 最后的 'O' 表示"无字符"占位符
```

车牌号 `0_0_22_27_27_33_16` 的解码过程:
- 第1位: provinces[0] = "皖"（省份）
- 第2位: alphabets[0] = "A"（字母）
- 第3-7位: ads[22]="M", ads[27]="2", ads[27]="2", ads[33]="8", ads[16]="Q"
- 完整车牌: **皖AM228Q**

### 1.5 CCPD 解析工具函数

将以下代码加入项目，用于解析 CCPD 文件名并提取标注信息:

```python
import os
import cv2
import numpy as np

def parse_ccpd_filename(filename):
    """解析 CCPD 文件名，提取标注信息"""
    name = os.path.splitext(filename)[0]
    parts = name.split('-')

    if len(parts) < 5:
        return None

    # 四角坐标
    box_coords = []
    for pt_str in parts[3].split('_'):
        x, y = pt_str.split('&')
        box_coords.append([int(x), int(y)])

    # 外接矩形
    rect_parts = parts[2].split('_')
    x1, y1 = rect_parts[0].split('&')
    x2, y2 = rect_parts[1].split('&')

    # 车牌号索引
    char_indices = list(map(int, parts[4].split('_')))

    # 字符索引对照表
    provinces = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
                 "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
                 "桂","琼","川","贵","云","藏","陕","甘","青","宁",
                 "新","警","学","O"]
    alphabets = ['A','B','C','D','E','F','G','H','J','K','L','M',
                 'N','P','Q','R','S','T','U','V','W','X','Y','Z','O']
    ads = ['A','B','C','D','E','F','G','H','J','K','L','M',
           'N','P','Q','R','S','T','U','V','W','X','Y','Z',
           '0','1','2','3','4','5','6','7','8','9','O']

    plate_text = provinces[char_indices[0]] + alphabets[char_indices[1]]
    for i in range(2, 7):
        if char_indices[i] != len(ads) - 1:  # 跳过 'O' 占位符
            plate_text += ads[char_indices[i]]

    return {
        "box_coords": np.array(box_coords, dtype=np.float32),  # 四角坐标
        "bbox": (int(x1), int(y1), int(x2), int(y2)),           # 外接矩形
        "plate_text": plate_text,                                # 车牌号
        "area_ratio": int(parts[0]),                             # 面积比
        "tilt": tuple(map(int, parts[1].split('_'))),            # 倾斜度
        "brightness": int(parts[5]) if len(parts) > 5 else 0,    # 亮度
        "blurriness": int(parts[6]) if len(parts) > 6 else 0,    # 模糊度
    }


def load_ccpd_subset(directory, max_samples=None):
    """加载 CCPD 子集目录"""
    images = []
    labels = []

    files = os.listdir(directory)
    if max_samples:
        files = files[:max_samples]

    for filename in files:
        if not filename.endswith('.jpg'):
            continue
        info = parse_ccpd_filename(filename)
        if info is None:
            continue
        images.append(os.path.join(directory, filename))
        labels.append(info)

    return images, labels
```

### 1.6 推荐使用方式

针对课程设计，建议如下使用 CCPD:

| 用途 | 子集 | 数量 | 说明 |
|------|------|------|------|
| 训练 SVM/CNN | CCPD-Base | 5000-10000 张 | 从 Base 子集中采样 |
| 验证集 | CCPD-Base | 1000 张 | 从 Base 子集中划分 |
| 测试集 (常规) | CCPD-Base | 2000 张 | 从 Base 子集中划分 |
| 测试集 (鲁棒性) | CCPD-Blur + Rotate | 各 500 张 | 测试系统鲁棒性 |
| 字符模板 | CCPD-Base | 100 张 | 截取清晰字符做模板 |

**数据集划分脚本**: CCPD 官方提供了 train/val/test 划分文件，位于 GitHub 仓库的 `split/` 目录下。

---

## 二、AOLP 数据集（备选）

### 2.1 简介

AOLP（Application-Oriented License Plate）是台湾地区发布的开源车牌数据集，适合作为补充测试数据。

- **GitHub**: https://github.com/AvLab-CV/AOLP
- **图像数**: 2049 张
- **开源协议**: 学术研究用途

### 2.2 子集说明

| 子集 | 说明 | 场景 |
|------|------|------|
| Access | 停车场出入口 | 闸机拍摄，光照较稳定 |
| Law Enforcement | 执法场景 | 违停抓拍，角度多样 |
| Road | 道路巡逻 | 巡逻车拍摄，运动模糊 |

### 2.3 注意事项

- 台湾车牌格式与大陆不同（如格式为 `ABC-1234`），字符集有差异
- 适合用于测试系统对非标准场景的泛化能力
- 不建议作为主训练集

---

## 三、自建数据集（补充方案）

### 3.1 数据采集

如果需要测试特定场景（如学校停车场），可以自己采集数据:

**拍摄建议**:
- 使用手机或相机在停车场、路边拍摄
- 拍摄距离: 3-10 米（模拟实际监控距离）
- 分辨率: 1080p 以上
- 覆盖不同光照: 白天、傍晚、夜间
- 覆盖不同角度: 正面、侧面、俯拍
- 覆盖不同车牌类型: 蓝牌、绿牌、黄牌
- 目标数量: 200-500 张即可起步

### 3.2 标注工具

| 工具 | 特点 | 推荐场景 |
|------|------|----------|
| **LabelImg** | 轻量级，离线使用，输出 PASCAL VOC XML | 快速标注矩形框 |
| **CVAT** | 在线工具，支持视频标注、多人协作 | 大规模标注 |
| **Roboflow** | 在线平台，自动增强、格式转换 | 一站式处理 |
| **Label Studio** | 开源，支持多种标注类型 | 综合项目 |

**标注内容**:
1. 车牌区域: 矩形框 (x, y, width, height)
2. 车牌号码: 文本标签
3. (可选) 字符位置: 每个字符的矩形框

### 3.3 数据增强

当数据量不足时，使用数据增强扩充训练集:

```python
import cv2
import numpy as np
import random

def augment_plate_image(image):
    """车牌图像数据增强"""
    augmented = []

    # 1. 旋转 (±15度)
    h, w = image.shape[:2]
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)

    # 2. 亮度调整
    for factor in [0.6, 0.8, 1.2, 1.5]:
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented.append(adjusted)

    # 3. 高斯噪声
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented.append(noisy)

    # 4. 高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    augmented.append(blurred)

    # 5. 水平翻转 (注意: 翻转后字符顺序会变，需调整标签)
    # flipped = cv2.flip(image, 1)
    # augmented.append(flipped)

    return augmented
```

---

## 四、RodoSol-ALPR 数据集（补充参考）

- **来源**: 巴西公开数据集
- **规模**: 20000 张图片
- **特点**: 包含完整的车牌检测和识别标注
- **适用**: 测试系统对国际车牌的扩展能力
- **获取**: https://github.com/ramonaysa/RodoSol-ALPR

---

## 五、数据集获取方案推荐

根据课程设计的时间和需求，推荐以下方案组合:

### 方案 A: 快速启动（推荐，1-2 天搞定）

```
CCPD-Base 采样 200 张 (测试) + 合成模板 (内置)
```

- 从 CCPD-Base 中随机采样 200 张图像用于测试
- 字符识别使用项目内置的合成模板（开箱即用）
- 适合时间紧张、只需演示功能的情况

### 方案 B: 标准课程设计（推荐，3-5 天）

```
CCPD-Base 5000 张 (训练 SVM) + 1000 张 (测试) + CCPD-Blur 200 张 (鲁棒性测试)
```

- 下载 CCPD2019 数据集
- 从 Base 子集采样 5000 张训练 SVM 字符识别模型
- 划分 1000 张作为测试集
- 使用 CCPD-Blur 子集测试系统鲁棒性
- 在报告中做不同条件的对比实验

### 方案 C: 完整实验（推荐，1 周以上）

```
CCPD 全子集 + CCPD-Green + 自建数据 + 数据增强
```

- 使用 CCPD 全部子集进行全面评估
- 训练 CNN 模型（需要 PyTorch + GPU）
- 自建小规模数据集测试实际场景
- 数据增强扩充训练集
- 在报告中做多方法、多数据集的完整对比

---

## 六、引用

如果使用 CCPD 数据集，请在报告中引用:

```bibtex
@inproceedings{xu2018towards,
  title={Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={255--271},
  year={2018}
}
```
