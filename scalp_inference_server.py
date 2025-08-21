from flask import Flask, request, jsonify
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image

# PyTorch 2.6+ 대응: 신뢰할 수 있는 클래스 등록
torch.serialization.add_safe_globals({'EfficientNet': EfficientNet})

app = Flask(__name__)

# Load all models
model_paths = {
    "미세각질": "model1_full.pt",
    "탈모": "model2_full.pt",
    "모낭사이홍반": "model3_full.pt",
    "모낭홍반농포": "model4_full.pt",
    "비듬": "model5_full.pt",
    "피지과다": "model6_full.pt",
    "모발밀도": "model7_full.pt"
}

models = {}
for name, path in model_paths.items():
    models[name] = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    models[name].eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize([600, 600]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/ai", methods=["POST"])
def ai():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    results = {}
    for disease, model in models.items():
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = torch.argmax(prob).item()
            results[disease] = {
                "class_index": pred_class,
                "confidence": round(prob[pred_class].item(), 3)
            }

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)






# from flask import Flask, request, jsonify
# import io
# import os
# import pathlib
# import requests
# import torch
# from efficientnet_pytorch import EfficientNet
# from torchvision import transforms
# from PIL import Image
# from typing import Dict
#
# # PyTorch 2.6+ 대응: 신뢰할 수 있는 클래스 등록
# torch.serialization.add_safe_globals({'EfficientNet': EfficientNet})
#
# app = Flask(__name__)
#
# # ---- 설정 ----
# MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", "./models")).resolve()
# MODEL_DIR.mkdir(parents=True, exist_ok=True)
#
# MODEL_URLS: Dict[str, str] = {
#     "미세각질": "https://smartbrushai.s3.us-east-1.amazonaws.com/model1_full.pt",
#     "탈모": "https://smartbrushai.s3.us-east-1.amazonaws.com/model2_full.pt",
#     "모낭사이홍반": "https://smartbrushai.s3.us-east-1.amazonaws.com/model3_full.pt",
#     "모낭홍반농포": "https://smartbrushai.s3.us-east-1.amazonaws.com/model4_full.pt",
#     "비듬": "https://smartbrushai.s3.us-east-1.amazonaws.com/model5_full.pt",
#     "피지과다": "https://smartbrushai.s3.us-east-1.amazonaws.com/model6_full.pt",
# }
#
# def download_if_needed(url: str, dest_dir: pathlib.Path) -> pathlib.Path:
#     """URL을 dest_dir로 다운로드(이미 존재하면 재사용)."""
#     filename = url.split("/")[-1]
#     dest_path = dest_dir / filename
#     if dest_path.exists() and dest_path.stat().st_size > 0:
#         return dest_path
#
#     # 스트리밍 다운로드
#     with requests.get(url, stream=True, timeout=60) as r:
#         r.raise_for_status()
#         tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
#         with open(tmp_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=1024 * 1024):
#                 if chunk:
#                     f.write(chunk)
#         tmp_path.replace(dest_path)
#     return dest_path
#
# # ---- 모델 로드 ----
# models: Dict[str, torch.nn.Module] = {}
# for name, url in MODEL_URLS.items():
#     local_path = download_if_needed(url, MODEL_DIR)
#     # PyTorch 2.6+ 에서 weights_only=False 필요 (피클된 전체 모델)
#     model = torch.load(local_path, map_location=torch.device('cpu'), weights_only=False)
#     model.eval()
#     models[name] = model
#
# # ---- 전처리 ----
# transform = transforms.Compose([
#     transforms.Resize([600, 600]),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
#
# @app.route("/ai", methods=["POST"])
# def ai():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400
#
#     image = Image.open(request.files['image']).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0)
#
#     results = {}
#     with torch.inference_mode():
#         for disease, model in models.items():
#             output = model(image_tensor)
#             prob = torch.nn.functional.softmax(output[0], dim=0)
#             pred_class = int(torch.argmax(prob).item())
#             results[disease] = {
#                 "class_index": pred_class,
#                 "confidence": round(float(prob[pred_class].item()), 3),
#             }
#
#     return jsonify(results)
#
# if __name__ == "__main__":
#     # 필요 시 호스트/포트 조정
#     app.run(host="0.0.0.0", port=5000)
