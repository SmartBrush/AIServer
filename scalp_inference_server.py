from flask import Flask, request, jsonify
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import numpy as np

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
    m = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    m.eval()
    models[name] = m

# Preprocessing
transform = transforms.Compose([
    transforms.Resize([600, 600]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/ai", methods=["POST"])
def ai():
    # 다중 파일 입력: image 필드로 여러 개 업로드
    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': 'No image provided'}), 400

    # 질환별 확률 누적
    sum_probs = {disease: None for disease in models.keys()}
    valid_count = 0

    with torch.inference_mode():
        for f in files:
            try:
                image = Image.open(f.stream).convert('RGB')
            except Exception:
                # 잘못된 이미지는 평균에서 제외
                continue

            x = transform(image).unsqueeze(0)     # [1, 3, H, W]

            for disease, model in models.items():
                logits = model(x)                  # [1, C]
                prob = torch.softmax(logits[0], dim=0).cpu().numpy()  # [C]

                if sum_probs[disease] is None:
                    sum_probs[disease] = prob.copy()
                else:
                    sum_probs[disease] += prob

            valid_count += 1

    if valid_count == 0:
        return jsonify({'error': 'All images were invalid'}), 400

    # 평균 확률 → 최종 결과
    results = {}
    for disease, s in sum_probs.items():
        mean_prob = s / float(valid_count)         # [C]
        pred = int(np.argmax(mean_prob))
        conf = float(mean_prob[pred])
        results[disease] = {
            "class_index": pred,
            "confidence": round(conf, 3)
        }

    return jsonify({
        "count": valid_count,   # 평균에 사용된 유효 이미지 수
        "results": results
    })

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000)
    app.run(host="0.0.0.0", port=8000)








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
