"""
app.py — FastAPI inference server for aerial image segmentation.

Endpoints:
  GET  /health       — Model metadata, device, uptime
  POST /predict      — Upload image → segmentation mask + overlay + class percentages
  GET  /visualize    — HTML demo page with inline segmentation overlay

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

Example curl commands:
    curl -X POST -F "file=@image.jpg" http://localhost:8000/predict -o result.zip
    curl http://localhost:8000/health
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.augmentation import get_transforms
from src.data.dataset import LoveDA
from src.models.efficient_unet import EfficientUNet
from src.models.vanilla_unet import VanillaUNet
from src.training.sliding_window import sliding_window_inference
from src.utils.config import load_config

app = FastAPI(title="Aerial Semantic Segmentation", version="1.0.0")
model = None
config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
class_names = LoveDA.CLASS_NAMES

# ── Request batching ────────────────────────────────────────────────────────────
# Collects individual inference requests and processes them as a batch for
# throughput optimisation on GPU. Configurable via environment variables:
#   BATCH_MAX_SIZE: max images per batch (default: 4)
#   BATCH_MAX_WAIT: max wait time in seconds (default: 0.1)


@dataclass
class BatchRequest:
    image_np: np.ndarray
    future: asyncio.Future


class BatchProcessor:
    def __init__(self, max_size: int = 4, max_wait: float = 0.1):
        self.max_size = max_size
        self.max_wait = max_wait
        self.queue: List[BatchRequest] = []
        self.lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def submit(self, image_np: np.ndarray) -> Dict:
        future = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append(BatchRequest(image_np, future))
            if len(self.queue) >= self.max_size:
                await self._flush()
        return await future

    async def _flush(self):
        async with self.lock:
            if not self.queue:
                return
            batch = self.queue[:]
            self.queue.clear()

        asyncio.create_task(self._process_batch(batch))

    async def _process_batch(self, batch: List[BatchRequest]):
        try:
            images = np.stack([r.image_np for r in batch], axis=0)
            transform = get_transforms("val", config)

            # Normalize all images
            tensors = []
            for img in images:
                aug = transform(image=img)
                t = torch.from_numpy(aug["image"]).permute(2, 0, 1).float()
                tensors.append(t)
            batch_tensor = torch.stack(tensors).to(device)

            # Batch inference
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                outputs = model(batch_tensor)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()

            for i, req in enumerate(batch):
                pred = preds[i]
                partial_pct = {}
                total = pred.size
                for idx, name in enumerate(class_names):
                    partial_pct[name] = round(int(np.sum(pred == idx)) / total * 100, 2)
                mask_rgb = LoveDA.decode_mask(pred)
                img_f = req.image_np.astype(np.float32) / 255.0
                overlay = (0.5 * img_f + 0.5 * (mask_rgb / 255.0))
                overlay = (overlay * 255).astype(np.uint8)
                _, m_bytes = cv2.imencode(".png", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
                _, o_bytes = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                req.future.set_result({
                    "mask_bytes": m_bytes.tobytes(),
                    "overlay_bytes": o_bytes.tobytes(),
                    "percentages": partial_pct,
                })
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def run(self):
        while True:
            await asyncio.sleep(self.max_wait)
            async with self.lock:
                if self.queue:
                    await self._flush()

    def start(self, loop):
        self._task = asyncio.ensure_future(self.run(), loop=loop)
        return self._task


batch_processor = BatchProcessor(
    max_size=int(os.environ.get("BATCH_MAX_SIZE", "4")),
    max_wait=float(os.environ.get("BATCH_MAX_WAIT", "0.1")),
)


def load_model_for_serving(checkpoint_path: str, cfg: Dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    arch = model_cfg.get("architecture", "efficient_unet")
    num_classes = cfg["dataset"]["num_classes"]

    if arch == "vanilla_unet":
        m = VanillaUNet(num_classes=num_classes)
    elif arch == "efficient_unet":
        m = EfficientUNet(
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b4"),
            encoder_weights=None,
            num_classes=num_classes,
            decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32, 16]),
            use_scse=model_cfg.get("use_scse", True),
            deep_supervision=False,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if arch == "efficient_unet":
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("aux_heads.")}

    m.load_state_dict(state_dict, strict=False)
    m = m.to(device)
    m.eval()
    return m


@app.on_event("startup")
async def startup_event():
    global model, config
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pth")
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")

    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
        )

    config = load_config(config_path)

    if os.path.exists(ckpt_path):
        model = load_model_for_serving(ckpt_path, config)
        print(f"Model loaded from {ckpt_path} | Device: {device}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found.")

    # Start background batch processor
    loop = asyncio.get_event_loop()
    batch_processor.start(loop)
    print(f"Batch processor started (max_size={batch_processor.max_size}, max_wait={batch_processor.max_wait}s)")


@app.get("/health")
async def health():
    status = {
        "status": "healthy" if model is not None else "degraded",
        "model_name": config["model"].get("architecture", "unknown") if config else "unknown",
        "num_classes": config["dataset"]["num_classes"] if config else 7,
        "class_names": class_names,
        "device": str(device),
        "uptime_seconds": int(time.time() - start_time),
    }
    return JSONResponse(status)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_json: bool = Query(False),
    patch_size: int = Query(512),
    stride: int = Query(256),
):
    if model is None:
        return JSONResponse({"error": "No model loaded."}, status_code=503)

    contents = await file.read()
    image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image_np is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_np.shape[:2]

    transform = get_transforms("val", config)
    augmented = transform(image=image_np)
    image_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()

    pred = sliding_window_inference(
        model=model, image=image_tensor,
        patch_size=patch_size, stride=stride,
        num_classes=config["dataset"]["num_classes"],
        gaussian_weight=True, device=device, mixed_precision=True,
    )

    total = pred.size
    percentages = {}
    for idx, name in enumerate(class_names):
        percentages[name] = round(int(np.sum(pred == idx)) / total * 100, 2)

    mask_rgb = LoveDA.decode_mask(pred)
    img_f = image_np.astype(np.float32) / 255.0
    overlay = (0.5 * img_f + 0.5 * (mask_rgb / 255.0))
    overlay = (overlay * 255).astype(np.uint8)

    _, mask_bytes = cv2.imencode(".png", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    _, overlay_bytes = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if return_json:
        return JSONResponse({
            "width": orig_w, "height": orig_h,
            "class_percentages": percentages,
        })

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mask.png", mask_bytes.tobytes())
        zf.writestr("overlay.png", overlay_bytes.tobytes())
        zf.writestr("results.json", json.dumps(
            {"width": orig_w, "height": orig_h, "class_percentages": percentages}, indent=2
        ))
    buf.seek(0)

    return Response(
        content=buf.read(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="segmentation_{uuid4().hex[:8]}.zip"'},
    )


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    use_batching: bool = Query(True, description="Use async batch processor for GPU throughput optimisation"),
    patch_size: int = Query(512),
    stride: int = Query(256),
):
    """Run batched segmentation inference on multiple uploaded images.

    When use_batching=True, images are accumulated and processed together
    on GPU for maximum throughput (configurable BATCH_MAX_SIZE and
    BATCH_MAX_WAIT via environment variables).

    Args:
        files: List of uploaded image files.
        use_batching: Whether to use the async batch processor.

    Returns:
        ZIP file containing individual results for each image.
    """
    if model is None:
        return JSONResponse({"error": "No model loaded."}, status_code=503)

    results = []
    for file in files:
        contents = await file.read()
        image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            results.append({"file": file.filename, "error": "Invalid image"})
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        if use_batching:
            batch_result = await batch_processor.submit(image_np)
            results.append({
                "file": file.filename,
                **batch_result,
            })
        else:
            orig_h, orig_w = image_np.shape[:2]
            transform = get_transforms("val", config)
            augmented = transform(image=image_np)
            img_t = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()

            pred = sliding_window_inference(
                model=model, image=img_t,
                patch_size=patch_size, stride=stride,
                num_classes=config["dataset"]["num_classes"],
                gaussian_weight=True, device=device, mixed_precision=True,
            )

            total = pred.size
            percentages = {}
            for idx, name in enumerate(class_names):
                percentages[name] = round(int(np.sum(pred == idx)) / total * 100, 2)

            mask_rgb = LoveDA.decode_mask(pred)
            img_f = image_np.astype(np.float32) / 255.0
            overlay = (0.5 * img_f + 0.5 * (mask_rgb / 255.0))
            overlay = (overlay * 255).astype(np.uint8)

            _, m_bytes = cv2.imencode(".png", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            _, o_bytes = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            results.append({
                "file": file.filename,
                "mask_bytes": m_bytes.tobytes(),
                "overlay_bytes": o_bytes.tobytes(),
                "percentages": percentages,
            })

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(results):
            if "error" in r:
                zf.writestr(f"{i}_{r['file']}/error.txt", r["error"])
            else:
                zf.writestr(f"{i}_{r['file']}/mask.png", r["mask_bytes"])
                zf.writestr(f"{i}_{r['file']}/overlay.png", r["overlay_bytes"])
                zf.writestr(f"{i}_{r['file']}/results.json", json.dumps(
                    {"class_percentages": r["percentages"]}, indent=2
                ))
    buf.seek(0)

    return Response(
        content=buf.read(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_segmentation_{uuid4().hex[:8]}.zip"'},
    )


HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Aerial Image Segmentation</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#f5f5f5}
h1{color:#333}.container{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
.upload-area{border:2px dashed #ccc;border-radius:8px;padding:40px;text-align:center;margin:20px 0}
.image-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}
.image-grid img{width:100%;border-radius:4px;border:1px solid #ddd}
.stats-table{width:100%;border-collapse:collapse;margin:20px 0}
.stats-table th,.stats-table td{padding:8px 12px;text-align:left;border-bottom:1px solid #ddd}
.stats-table th{background:#4CAF50;color:white}
.btn{background:#4CAF50;color:white;padding:10px 24px;border:none;border-radius:4px;cursor:pointer;font-size:16px}
.btn:hover{background:#45a049}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:10px 0;font-size:13px}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-color{width:12px;height:12px;border-radius:2px;display:inline-block}
#spinner{display:none;margin:20px auto;border:4px solid #f3f3f3;border-top:4px solid #4CAF50;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
</style></head>
<body>
<h1>Aerial Image Semantic Segmentation</h1>
<div class="container">
<div class="upload-area" id="dropZone">
<p style="font-size:18px;color:#666;">Drop an aerial image here</p>
<p style="color:#999;">or</p>
<input type="file" id="fileInput" accept="image/png,image/jpeg" hidden>
<button class="btn" onclick="document.getElementById('fileInput').click()">Choose Image</button>
</div>
<div id="spinner"></div>
<div id="results" style="display:none;">
<div class="image-grid">
<div><strong>Original</strong><br><img id="originalImg"></div>
<div><strong>Overlay</strong><br><img id="overlayImg"></div>
</div>
<h3>Class Distribution</h3>
<div class="legend" id="legend"></div>
<table class="stats-table"><thead><tr><th>Class</th><th>%</th></tr></thead><tbody id="statsBody"></tbody></table>
</div></div>
<script>
const cols=['#000','#00F','#0FF','#F00','#0F0','#808080','#FF0'];
const names=['Background','Building','Road','Water','Barren','Forest','Agriculture'];
const dz=document.getElementById('dropZone'),fi=document.getElementById('fileInput');
const sp=document.getElementById('spinner'),res=document.getElementById('results');
['dragenter','dragover','dragleave','drop'].forEach(e=>dz.addEventListener(e,e=>e.preventDefault()));
dz.addEventListener('drop',e=>{if(e.dataTransfer.files[0])process(e.dataTransfer.files[0])});
fi.addEventListener('change',e=>{if(e.target.files[0])process(e.target.files[0])});
function process(f){if(!f.type.startsWith('image/'))return;res.style.display='none';sp.style.display='block';
const fd=new FormData();fd.append('file',f);const r=new FileReader();
r.onload=e=>document.getElementById('originalImg').src=e.target.result;r.readAsDataURL(f);
fetch('/predict',{method:'POST',body:fd}).then(r=>r.blob()).then(b=>{
var zip=new JSZip();return zip.loadAsync(b);
}).then(z=>Promise.all([z.file('overlay.png').async('base64'),z.file('results.json').async('string')])
).then(([ob64,rj])=>{
document.getElementById('overlayImg').src='data:image/png;base64,'+ob64;
const d=JSON.parse(rj);const leg=document.getElementById('legend');
leg.innerHTML=names.map((n,i)=>'<span class="legend-item"><span class="legend-color" style="background:'+cols[i]+'"></span>'+n+'</span>').join('');
const tb=document.getElementById('statsBody');
tb.innerHTML=names.map((n,i)=>{const p=d.class_percentages[n]||0;
return'<tr><td><span class="legend-color" style="background:'+cols[i]+';vertical-align:middle"></span> '+n+'</td><td>'+p+'%</td></tr>'}).join('');
sp.style.display='none';res.style.display='block';
}).catch(e=>{sp.style.display='none';alert('Error:'+e.message)})}
</script><script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
</body></html>"""


@app.get("/visualize", response_class=HTMLResponse)
async def visualize():
    return HTMLResponse(HTML_PAGE)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    uvicorn.run(app, host=args.host, port=args.port)
