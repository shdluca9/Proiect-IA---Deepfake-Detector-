import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- CONFIGURARE ---
# Aici pui calea exactă unde s-au descărcat videoclipurile tale
# Verifică în PyCharm structura folderelor tale!
PATH_REAL_VIDEOS = "./Date_FF/original_sequences/youtube/c23/videos"
PATH_FAKE_VIDEOS = "./Date_FF/manipulated_sequences/Deepfakes/c23/videos"

# Aici se vor salva fețele decupate
OUTPUT_FOLDER = "./Data_Processed"

# Câte cadre să sărim? (30 cadre = aprox 1 secundă).
# Luăm 1 cadru pe secundă ca să avem diversitate.
FRAME_SKIP = 30
# Maximum de imagini per video (ca să nu umplem hardul degeaba)
MAX_FACES_PER_VIDEO = 20


def setup_directories():
    # Creăm folderele de ieșire dacă nu există
    os.makedirs(f"{OUTPUT_FOLDER}/Real", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/Fake", exist_ok=True)


def process_folder(input_path, label_name):
    print(f"\n--- Procesez videoclipurile din: {label_name} ---")

    # Verificăm dacă folderul sursă există
    if not os.path.exists(input_path):
        print(f"EROARE: Nu găsesc folderul: {input_path}")
        print("Verifică calea din variabila PATH_REAL_VIDEOS sau PATH_FAKE_VIDEOS")
        return

    # Pregătim detectorul de fețe (MTCNN)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Folosesc dispozitivul: {device}")

    # margin=0 adaugă puțin spațiu în jurul feței, select_largest=True ia doar fața principală
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, select_largest=True, device=device,
                  post_process=False)

    video_files = [f for f in os.listdir(input_path) if f.endswith('.mp4')]

    for video_name in tqdm(video_files):
        video_path = os.path.join(input_path, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        faces_saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesăm doar un cadru la fiecare FRAME_SKIP (ex: la fiecare 30 de cadre)
            if frame_count % FRAME_SKIP == 0:
                # Convertim din BGR (OpenCV) în RGB (formatul corect pentru AI)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                frame_pil = Image.fromarray(frame)

                # Numele imaginii salvate
                save_path = f"{OUTPUT_FOLDER}/{label_name}/{video_name.split('.')[0]}_frame{frame_count}.jpg"

                # MTCNN detectează și salvează automat fața dacă o găsește
                mtcnn(frame_pil, save_path=save_path)

                # Verificăm dacă s-a salvat fișierul (dacă mtcnn a găsit față)
                if os.path.exists(save_path):
                    faces_saved += 1

            frame_count += 1

            # Dacă am strâns destule fețe din acest video, trecem la următorul
            if faces_saved >= MAX_FACES_PER_VIDEO:
                break

        cap.release()


if __name__ == "__main__":
    setup_directories()

    # 1. Procesăm videoclipurile REALE
    process_folder(PATH_REAL_VIDEOS, "Real")

    # 2. Procesăm videoclipurile FAKE
    process_folder(PATH_FAKE_VIDEOS, "Fake")

    print("\n GATA! Verifică folderul 'Data_Processed'.")