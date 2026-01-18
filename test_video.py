import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import os
import shutil
import numpy as np

# --- CONFIGURARE ---
MODEL_PATH = "deepfake_model.pth"
VIDEO_PATH = "video_de_test.mp4"
DEBUG_FOLDER = "debug_faces"


def load_trained_model():
    print(f"[INFO] Configurez dispozitivul...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Incarc arhitectura EfficientNet...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("[SUCCES] Model incarcat!")
    except Exception as e:
        print(f"[EROARE] Nu pot incarca modelul: {e}")
        return None, None

    model = model.to(device)
    model.eval()
    return model, device


def predict_video(model, device, video_path):
    # Curatenie folder debug
    if os.path.exists(DEBUG_FOLDER):
        shutil.rmtree(DEBUG_FOLDER)
    os.makedirs(DEBUG_FOLDER)

    print(f"[INFO] Analizez video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "[EROARE] Nu pot deschide video-ul."

    # MTCNN setat sa ne dea doar coordonatele (fara post-procesare automata)
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    predictions = []
    frames_processed = 0
    faces_found = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analizam 1 cadru la fiecare 15
        if frames_processed % 15 == 0:

            # 1. CONVERSIE BGR -> RGB (Manual si Explicit)
            # Acesta este pasul critic pentru culori!
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
            except:
                continue

            # 2. DETECTIE FATA (Doar coordonatele)
            try:
                boxes, _ = mtcnn.detect(pil_img)

                if boxes is not None:
                    # Luam coordonatele primei fete
                    box = boxes[0]

                    # 3. DECUPARE MANUALA (CROP)
                    # Decupam direct din imaginea RGB, deci imposibil sa iasa albastra
                    face_img = pil_img.crop(box)

                    faces_found += 1

                    # 4. PREDICTIE
                    # Pregatim pentru AI
                    face_tensor = trans(face_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        fake_prob = probs[0][0].item()
                        predictions.append(fake_prob)

                    # 5. SALVARE DEBUG
                    # Salvam exact ce am decupat manual
                    save_name = f"{DEBUG_FOLDER}/frame_{frames_processed}_score_{int(fake_prob * 100)}.jpg"
                    face_img.save(save_name)

                    if faces_found % 10 == 0:
                        print(f"... Am analizat {faces_found} fete.")

            except Exception as e:
                pass

        frames_processed += 1

    cap.release()

    if not predictions:
        return "[REZULTAT] Nu s-au detectat fete."

    # --- CALCUL VERDICT ---
    predictions.sort(reverse=True)
    top_k = min(10, len(predictions))
    top_scores = predictions[:top_k]
    avg_top_score = sum(top_scores) / top_k

    print(f"\n--- REZULTATE FINALE ---")
    print(f"Top 5 scoruri de Fake detectate: {[f'{p:.2f}' for p in top_scores[:5]]}")

    fake_percent = avg_top_score * 100

    if avg_top_score > 0.65:
        return f"VERDICT: DEEPFAKE (Probabilitate: {fake_percent:.1f}%)"
    elif avg_top_score < 0.35:
        return f"VERDICT: REAL (Probabilitate Fake: {fake_percent:.1f}%)"
    else:
        return f"VERDICT: INCERT (Scor: {fake_percent:.1f}%)"


if __name__ == "__main__":
    model, device = load_trained_model()
    if model and os.path.exists(VIDEO_PATH):
        print(predict_video(model, device, VIDEO_PATH))
        print(f"\nTe rog verifica ACUM folderul '{DEBUG_FOLDER}'. Fețele trebuie sa aiba culoarea pielii!")
    else:
        print("Verifica daca ai fisierul video si modelul .pth!")