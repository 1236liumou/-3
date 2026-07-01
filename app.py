"""
Streamlit 可视化界面 — 车牌识别系统
双引擎架构: HyperLPR3 AI 引擎 + 传统 CV 五阶段

运行方式:
    streamlit run app.py
"""

import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from lpr.pipeline import LPRPipeline
from lpr.config import APP_CONFIG

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
)

# ============================================================
# 自定义样式
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .engine-badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ai {
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    .badge-cv {
        background: #e3f2fd;
        color: #1565c0;
        border: 1px solid #2196f3;
    }
    .plate-result {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: 4px;
        text-align: center;
        padding: 12px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        margin: 8px 0;
    }
    .timing-bar {
        background: #e0e0e0;
        border-radius: 4px;
        height: 22px;
        overflow: hidden;
        margin: 3px 0;
    }
    .timing-fill {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 工具函数
# ============================================================

def decode_upload_image(uploaded_file):
    """解码上传图片，cv2 失败时用 PIL 兜底"""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    # PIL 兜底
    uploaded_file.seek(0)
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def cv2_to_pil(cv2_image):
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def display_timing_bar(timing_dict, total_time):
    if not timing_dict:
        return
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    for i, (stage, t) in enumerate(timing_dict.items()):
        pct = (t / total_time * 100) if total_time > 0 else 0
        color = colors[i % len(colors)]
        st.markdown(f"**{stage}**: {t:.2f} ms ({pct:.1f}%)")
        st.markdown(
            f'<div class="timing-bar">'
            f'<div class="timing-fill" style="width: {pct}%; background: {color}">'
            f'{t:.1f}ms</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 缓存流水线
# ============================================================

@st.cache_resource
def get_pipeline(engine, locate_method="multi", segment_method="projection",
                 recognize_method="template"):
    return LPRPipeline(
        engine=engine,
        locate_method=locate_method,
        segment_method=segment_method,
        recognize_method=recognize_method,
        verbose=True,
    )


# ============================================================
# 主界面
# ============================================================

st.markdown('<div class="main-header">车牌识别系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">数字图像处理课程设计 · '
    '双引擎架构: AI + 传统CV</div>',
    unsafe_allow_html=True,
)

# ============================================================
# 侧边栏
# ============================================================

st.sidebar.markdown("### 引擎选择")

engine = st.sidebar.radio(
    "识别引擎",
    ["hyperlpr", "traditional"],
    format_func=lambda x: {
        "hyperlpr": "AI 引擎 (HyperLPR3) — 高精度",
        "traditional": "传统 CV 五阶段 — 课程演示",
    }[x],
    help="AI 引擎使用深度学习模型，准确率高；传统CV展示图像处理算法原理",
)

if engine == "traditional":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 传统 CV 参数")

    locate_method = st.sidebar.selectbox(
        "车牌定位方法",
        ["multi", "color", "edge"],
        format_func=lambda x: {
            "multi": "多颜色联合 (蓝+绿+黄)",
            "color": "颜色过滤法 (仅蓝牌)",
            "edge": "边缘检测法",
        }[x],
    )
    segment_method = st.sidebar.selectbox(
        "字符分割方法",
        ["projection", "connected"],
        format_func=lambda x: {
            "projection": "垂直投影法",
            "connected": "连通域分析法",
        }[x],
    )
    recognize_method = st.sidebar.selectbox(
        "字符识别方法",
        ["template", "svm", "cnn"],
        format_func=lambda x: {
            "template": "模板匹配 (无需训练)",
            "svm": "SVM 支持向量机",
            "cnn": "CNN 卷积神经网络",
        }[x],
    )
else:
    locate_method = "multi"
    segment_method = "projection"
    recognize_method = "template"

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
badge_class = "badge-ai" if engine == "hyperlpr" else "badge-cv"
badge_text = "AI 引擎" if engine == "hyperlpr" else "传统 CV"
st.sidebar.markdown(
    f'<span class="engine-badge {badge_class}">{badge_text}</span>',
    unsafe_allow_html=True,
)
st.sidebar.info(
    "**AI 引擎**: HyperLPR3 深度学习模型，端到端识别\n\n"
    "**传统 CV**: 预处理 → 定位 → 校正 → 分割 → 识别\n\n"
    "五阶段完整展示数字图像处理核心算法"
)

# ============================================================
# 获取流水线
# ============================================================

pipeline = get_pipeline(engine, locate_method, segment_method, recognize_method)

# ============================================================
# 图片上传
# ============================================================

uploaded_file = st.file_uploader(
    "上传车辆图片",
    type=APP_CONFIG["supported_formats"],
    help=f"支持 {', '.join(APP_CONFIG['supported_formats'])} 格式",
)

col1, col2 = st.columns([1, 1])

image = None
with col1:
    st.markdown("### 输入图像")
    if uploaded_file is not None:
        image = decode_upload_image(uploaded_file)
        if image is not None:
            st.image(cv2_to_pil(image), use_container_width=True)
        else:
            st.error("无法读取图片，请检查文件格式")
    else:
        st.info("请上传一张包含车牌的图片")

# ============================================================
# 识别按钮
# ============================================================

if uploaded_file is not None and image is not None:
    if st.button("开始识别", type="primary", use_container_width=True):

        with st.spinner("正在识别车牌..."):
            result = pipeline.recognize(image)

        with col2:
            st.markdown("### 识别结果")

            if result["success"]:
                # 结果图
                if result["result_image"] is not None:
                    st.image(cv2_to_pil(result["result_image"]),
                             use_container_width=True)

                st.markdown("---")

                # 车牌号大字显示
                plate_text = result["plate_text"] or "未识别"
                st.markdown(
                    f'<div class="plate-result">{plate_text}</div>',
                    unsafe_allow_html=True,
                )

                # 核心指标
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("置信度", f"{result['confidence']:.1%}")
                with m2:
                    st.metric("总耗时", f"{result['total_time']:.0f}ms")
                with m3:
                    engine_label = "AI" if result["engine"] == "hyperlpr" else "CV"
                    st.metric("引擎", engine_label)

                # 格式校验
                if result["is_valid"]:
                    st.success("车牌格式校验通过")
                else:
                    st.warning("车牌格式异常 (可能需要调优参数)")

                # 多车牌检测 (HyperLPR)
                if result.get("all_plates") and len(result["all_plates"]) > 1:
                    st.info(f"检测到 {len(result['all_plates'])} 个车牌:")
                    for i, p in enumerate(result["all_plates"]):
                        st.markdown(
                            f"  车牌{i+1}: **{p['plate_text']}** "
                            f"(置信度: {p['confidence']:.1%})"
                        )

            else:
                st.error(f"识别失败: {result.get('error', '未知错误')}")

        # ============================================================
        # 中间结果展示
        # ============================================================

        if result["success"]:
            st.markdown("---")

            if engine == "hyperlpr":
                # AI 引擎展示
                st.markdown("### AI 引擎分析")

                tab1, tab2 = st.tabs(["车牌区域", "性能分析"])

                with tab1:
                    if result["plate_image"] is not None:
                        st.markdown("**截取的车牌区域:**")
                        st.image(cv2_to_pil(result["plate_image"]),
                                 use_container_width=True)

                    stage = result["stage_details"].get("hyperlpr", {})
                    st.markdown(f"**检测到车牌数**: {stage.get('num_detected', 0)}")
                    st.markdown(
                        f"**引擎耗时**: {stage.get('timing', 0):.2f} ms"
                    )

                with tab2:
                    timing = result["timing_summary"]
                    total = result["total_time"]
                    display_timing_bar(timing, total)
                    st.markdown(f"**总耗时**: {total:.1f} ms")
                    if total > 0:
                        st.markdown(f"**处理帧率**: {1000/total:.1f} FPS")

            else:
                # 传统 CV 五阶段展示
                st.markdown("### 传统 CV 五阶段中间结果")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "1. 预处理", "2. 车牌定位", "3. 倾斜校正",
                    "4. 字符分割", "5. 字符识别",
                ])

                # Tab 1: 预处理
                with tab1:
                    stage = result["stage_details"].get("preprocess", {})
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**灰度图**")
                        gray = pipeline.preprocessor.grayscale(image)
                        st.image(cv2_to_pil(gray), use_container_width=True)
                    with cols[1]:
                        st.markdown("**双边滤波**")
                        bilat = pipeline.preprocessor.bilateral_filter(gray)
                        st.image(cv2_to_pil(bilat), use_container_width=True)
                    with cols[2]:
                        st.markdown("**CLAHE增强**")
                        clahe = pipeline.preprocessor.clahe_enhance(bilat)
                        st.image(cv2_to_pil(clahe), use_container_width=True)
                    t = stage.get("timing", {})
                    st.markdown(
                        f"耗时: 灰度化 {t.get('grayscale', 0):.2f}ms · "
                        f"双边滤波 {t.get('bilateral_filter', 0):.2f}ms · "
                        f"CLAHE {t.get('clahe_enhance', 0):.2f}ms"
                    )

                # Tab 2: 车牌定位
                with tab2:
                    stage = result["stage_details"].get("locate", {})
                    st.markdown(f"**定位方法**: {stage.get('method', 'N/A')}")
                    st.markdown(
                        f"**候选区域数**: {stage.get('num_candidates', 0)}"
                    )
                    region = stage.get("plate_region", {})
                    if region:
                        st.markdown(f"**车牌区域**: {region.get('rect', 'N/A')}")
                        st.markdown(f"**评分**: {region.get('score', 0):.3f}")
                        st.markdown(f"**面积**: {region.get('area', 0)}")
                        st.markdown(
                            f"**宽高比**: {region.get('aspect_ratio', 0):.2f}"
                        )
                    if result["plate_image"] is not None:
                        st.markdown("**截取的车牌区域:**")
                        st.image(cv2_to_pil(result["plate_image"]),
                                 use_container_width=True)

                # Tab 3: 倾斜校正
                with tab3:
                    stage = result["stage_details"].get("skew_correction", {})
                    c = st.columns(2)
                    with c[0]:
                        st.markdown("**校正前**")
                        st.image(cv2_to_pil(result["plate_image"]),
                                 use_container_width=True)
                    with c[1]:
                        st.markdown("**校正后**")
                        if result["corrected_image"] is not None:
                            st.image(cv2_to_pil(result["corrected_image"]),
                                     use_container_width=True)
                    st.markdown(
                        f"**校正角度**: {stage.get('skew_angle', 0):.2f}"
                    )

                # Tab 4: 字符分割
                with tab4:
                    stage = result["stage_details"].get("segment", {})
                    st.markdown(f"**分割方法**: {stage.get('method', 'N/A')}")
                    method_used = stage.get("method_used", "")
                    if method_used and method_used != stage.get("method"):
                        st.info(f"实际使用: {method_used} (自动 fallback)")
                    st.markdown(f"**分割字符数**: {stage.get('num_chars', 0)}")
                    if result["char_images"]:
                        st.markdown("**分割结果:**")
                        n = min(len(result["char_images"]), 7)
                        cc = st.columns(n)
                        for i, char_img in enumerate(result["char_images"]):
                            if i < n:
                                with cc[i]:
                                    st.markdown(f"位置{i+1}")
                                    enlarged = cv2.resize(
                                        char_img, (80, 112),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                    st.image(cv2_to_pil(enlarged),
                                             use_container_width=True)
                    else:
                        st.warning("未分割出字符")

                # Tab 5: 字符识别
                with tab5:
                    stage = result["stage_details"].get("recognize", {})
                    st.markdown(f"**识别方法**: {stage.get('method', 'N/A')}")
                    char_details = stage.get("char_details", [])
                    if char_details:
                        st.markdown("**逐字符识别结果:**")
                        for detail in char_details:
                            pos = detail["position"]
                            char = detail["char"]
                            conf = detail["confidence"]
                            t_ms = detail["timing"]
                            if conf >= 0.7:
                                c = "green"
                            elif conf >= 0.4:
                                c = "orange"
                            else:
                                c = "red"
                            st.markdown(
                                f"位置{pos+1}: **{char}** · "
                                f"置信度 :{c}[{conf:.1%}] · "
                                f"耗时 {t_ms:.2f}ms"
                            )
                        st.markdown("---")
                        st.markdown(
                            f"**平均置信度**: {result['confidence']:.1%}"
                        )

            # ============================================================
            # 性能分析
            # ============================================================
            st.markdown("---")
            st.markdown("### 性能分析")
            timing = result["timing_summary"]
            total = result["total_time"]
            display_timing_bar(timing, total)
            st.markdown(f"**总耗时**: {total:.1f} ms")
            if total > 0:
                st.markdown(f"**处理帧率**: {1000/total:.1f} FPS")

else:
    with col2:
        st.markdown("### 识别结果")
        st.info("上传图片后点击「开始识别」")
