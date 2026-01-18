from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import yt_dlp
import shutil
import numpy as np

# --- 1. IMPORT MODUL AUDIO ---
try:
    from audio import verifica as verifica_audio

    print("[SERVER] Modulul Audio incarcat cu succes!")
except ImportError as e:
    print(f"[SERVER] ATENTIE: Modulul audio nu merge. Eroare: {e}")
    verifica_audio = None

# --- 2. IMPORT MODUL URL ---
try:
    from url_detector import analyze_url

    print("[SERVER] Modulul URL incarcat cu succes!")
except ImportError as e:
    print(f"[SERVER] ATENTIE: Modulul URL nu merge. Eroare: {e}")
    analyze_url = None

app = Flask(__name__)
CORS(app)

# --- CONFIGURARE VIDEO ---
MODEL_PATH = "deepfake_model.pth"
DOWNLOAD_FOLDER = "downloads"

if os.path.exists(DOWNLOAD_FOLDER):
    shutil.rmtree(DOWNLOAD_FOLDER)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# --- INCARCARE MODEL VIDEO ---
print("[SERVER] Incarc modelul VIDEO...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[SERVER] Model Video OK!")
except Exception as e:
    print(f"[SERVER] EROARE MODEL VIDEO: {e}")

mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def download_youtube_video(url):
    filename = os.path.join(DOWNLOAD_FOLDER, "temp_video.mp4")
    if os.path.exists(filename): os.remove(filename)

    ydl_opts = {
        'format': '18/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
        'noplaylist': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            return filename
    except Exception as e:
        print(f"Eroare download video: {e}")
    return None


def analyze_video_logic(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    predictions = []
    frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frames % 30 == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                boxes, _ = mtcnn.detect(pil_img)
                if boxes is not None:
                    face_img = pil_img.crop(boxes[0])
                    face_tensor = trans(face_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(face_tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        predictions.append(probs[0][0].item())
            except:
                pass
        frames += 1
        if len(predictions) > 40: break
    cap.release()

    if not predictions: return 0
    predictions.sort(reverse=True)
    top_k = min(10, len(predictions))
    avg = sum(predictions[:top_k]) / top_k
    return round(avg * 100, 2)


@app.route('/detect', methods=['POST'])
def detect_combined():
    data = request.get_json()
    url = data.get('url')
    if not url: return jsonify({"error": "No URL"}), 400

    print(f"\n[SERVER] Analizez URL: {url}")

    # 1. VIDEO
    print(">>> Rulez Analiza VIDEO...")
    video_path = download_youtube_video(url)
    video_score = 0
    if video_path:
        video_score = analyze_video_logic(video_path)
        try:
            os.remove(video_path)
        except:
            pass
    print(f"Resultat Video: {video_score}% Fake")

    # 2. AUDIO
    print(">>> Rulez Analiza AUDIO...")
    audio_score = 0
    if verifica_audio:
        try:
            _, prob_val = verifica_audio(url)
            audio_score = round(prob_val * 100, 2)
        except:
            audio_score = 0
    print(f"Resultat Audio: {audio_score}% Fake")

    # 3. URL CHECK
    print(">>> Rulez Analiza URL...")
    url_score = 0
    if analyze_url:
        try:
            url_score = analyze_url(url)
        except:
            url_score = 0
    print(f"Resultat URL: {url_score}% Risc")

    # 4. SCOR FINAL (MAXIMUL DINTRE CELE 3)
    final_score = max(video_score, audio_score, url_score)

    overall_verdict = "REAL"
    if final_score > 80:
        overall_verdict = "FAKE"
    elif final_score > 45:
        overall_verdict = "INCERT"

    response = {
        "result": overall_verdict,
        "final_score": final_score,
        "details": {
            "video_score": video_score,
            "audio_score": audio_score,
            "url_score": url_score
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)