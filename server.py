"""
车牌识别系统 — FastAPI 后端服务
双引擎架构: HyperLPR3 AI 引擎 + 传统 CV 五阶段

启动方式:
    cd 车牌识别系统
    python server.py

默认端口: 8502
"""
import io
import os
import sys
import base64
import traceback

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lpr.pipeline import LPRPipeline
from lpr.config import APP_CONFIG

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="车牌识别系统 API",
    description="双引擎车牌识别: HyperLPR3 AI + 传统CV",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录
WEBSITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "website")

# ============================================================
# 工具函数
# ============================================================
def cv2_to_base64(cv2_img, fmt=".jpg", quality=85):
    """numpy 图像 -> base64 字符串"""
    if cv2_img is None:
        return None
    if len(cv2_img.shape) == 2:
        # 灰度图转 BGR
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(fmt, cv2_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def decode_upload(file_bytes):
    """解码上传字节 -> numpy 图像"""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # PIL 兜底
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def serialize_result(result):
    """将 pipeline 结果中的 numpy 图像转为 base64"""
    serialized = {
        "success": result.get("success", False),
        "plate_text": result.get("plate_text", ""),
        "confidence": result.get("confidence", 0.0),
        "is_valid": result.get("is_valid", False),
        "engine": result.get("engine", ""),
        "total_time": round(result.get("total_time", 0), 1),
        "error": result.get("error", ""),
        "all_plates": result.get("all_plates", []),
        "timing_summary": {
            k: round(v, 1) for k, v in result.get("timing_summary", {}).items()
        },
        "stage_details": {},
    }

    # 图像序列化
    serialized["result_image"] = cv2_to_base64(result.get("result_image"))
    serialized["plate_image"] = cv2_to_base64(result.get("plate_image"))
    serialized["corrected_image"] = cv2_to_base64(result.get("corrected_image"))

    char_imgs = result.get("char_images")
    serialized["char_images"] = [
        cv2_to_base64(c, fmt=".png") for c in char_imgs
    ] if char_imgs else []

    # 阶段详情 (递归序列化)
    sd = result.get("stage_details", {})
    for stage_key, stage_val in sd.items():
        if isinstance(stage_val, dict):
            serialized["stage_details"][stage_key] = {
                k: (cv2_to_base64(v) if isinstance(v, np.ndarray) else v)
                for k, v in stage_val.items()
            }
            # timing 子字典四舍五入
            if "timing" in serialized["stage_details"][stage_key]:
                t = serialized["stage_details"][stage_key]["timing"]
                if isinstance(t, dict):
                    serialized["stage_details"][stage_key]["timing"] = {
                        k: round(v, 1) for k, v in t.items()
                    }
        else:
            serialized["stage_details"][stage_key] = str(stage_val)

    # char_details 处理
    if "recognize" in serialized["stage_details"]:
        cd = serialized["stage_details"]["recognize"].get("char_details", [])
        if cd:
            for item in cd:
                if "confidence" in item and isinstance(item["confidence"], float):
                    item["confidence"] = round(item["confidence"], 4)
                if "timing" in item and isinstance(item["timing"], float):
                    item["timing"] = round(item["timing"], 1)

    return serialized


# ============================================================
# 缓存
# ============================================================
_pipeline_cache = {}

def get_pipeline(engine, locate_method, segment_method, recognize_method):
    key = (engine, locate_method, segment_method, recognize_method)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = LPRPipeline(
            engine=engine,
            locate_method=locate_method,
            segment_method=segment_method,
            recognize_method=recognize_method,
            verbose=True,
        )
    return _pipeline_cache[key]


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "车牌识别系统 v2.0"}


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    engine: str = Form("hyperlpr"),
    locate_method: str = Form("multi"),
    segment_method: str = Form("projection"),
    recognize_method: str = Form("template"),
):
    """
    上传图片，返回识别结果 (含 base64 编码的中间图像)
    """
    try:
        # 读取上传文件
        file_bytes = await file.read()
        image = decode_upload(file_bytes)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "无法解码图片，请检查格式"},
            )

        # 获取流水线并识别
        pipeline = get_pipeline(engine, locate_method, segment_method, recognize_method)
        result = pipeline.recognize(image)

        # 序列化 (numpy -> base64)
        serialized = serialize_result(result)
        serialized["filename"] = file.filename

        return serialized

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"服务器错误: {str(e)}"},
        )


# ============================================================
# 静态页面服务
# ============================================================

@app.get("/")
def index():
    return FileResponse(os.path.join(WEBSITE_DIR, "index.html"))


# 挂载静态文件
if os.path.isdir(WEBSITE_DIR):
    app.mount("/static", StaticFiles(directory=WEBSITE_DIR), name="static")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    print(f"""
╔══════════════════════════════════════════════╗
║      车牌识别系统 v2.0 — FastAPI 后端         ║
║                                              ║
║  Web 界面: http://localhost:{port}              ║
║  API 文档: http://localhost:{port}/docs        ║
║  Health:   http://localhost:{port}/api/health  ║
╚══════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
