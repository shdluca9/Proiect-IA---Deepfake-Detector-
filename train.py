import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- CONFIGURARE ---
DATA_DIR = "./Data_Processed"  # Folderul cu poze
MODEL_SAVE_PATH = "deepfake_model.pth"
BATCH_SIZE = 16
EPOCHS = 5  # Am crescut la 5 epoci pentru rezultate mai bune
LEARNING_RATE = 0.001


def train_model():
    # --- AICI AM ADAUGAT AUGMENTAREA (Rotiri, Culori, etc.) ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),

        # Trucuri ca să învețe mai bine (Data Augmentation):
        transforms.RandomHorizontalFlip(p=0.5),  # Oglindire
        transforms.RandomRotation(degrees=15),  # Rotire ușoară
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Schimbare lumina

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Incarc imaginile pentru antrenare...")
    if not os.path.exists(DATA_DIR):
        print(f"EROARE: Nu gasesc folderul {DATA_DIR}. Ruleaza intai process_data.py!")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    print(f"Clase detectate: {full_dataset.class_to_idx}")

    # Împărțim datele
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Pregătim modelul
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Antrenamentul va rula pe: {device}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Start Antrenare (Noua versiune cu Augmentare) ---")

    for epoch in range(EPOCHS):
        print(f"Epoca {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Salvarea finală
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel salvat cu succes în '{MODEL_SAVE_PATH}'")
    print("Acum poti rula test_video.py!")


if __name__ == "__main__":
    train_model()