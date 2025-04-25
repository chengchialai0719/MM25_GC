import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from tqdm import tqdm
from utils import HallucinationDataset

# Configuration
JSON_PATH = '../dataset/Hallucination Detection/json/train.json'
IMAGE_FOLDER = '../dataset/Hallucination Detection/images'
BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load CLIP model and preprocessing
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()  # or model.train() if you plan to finetune

    # Load tokenizer and dataset
    tokenizer = clip.tokenize  # From OpenAI CLIP
    dataset = HallucinationDataset(JSON_PATH, IMAGE_FOLDER, tokenizer, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optional: classification head
    classifier = nn.Linear(model.text_projection.shape[1], 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    model.eval()  # Keep CLIP frozen unless you're fine-tuning

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        correct_total = 0
        total_samples = 0

        for images, tokenized_texts, labels, correct_ids in tqdm(loader):
            images = images.to(DEVICE)
            batch_size = images.size(0)

            # Reshape text input: [B, 4, 77] → [B*4, 77]
            tokenized_texts = tokenized_texts.view(-1, tokenized_texts.size(-1)).to(DEVICE)

            labels = labels.to(DEVICE)
            correct_ids = correct_ids.to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(images)  # [B, 512]
                text_features = model.encode_text(tokenized_texts)  # [B*4, 512]

            text_features = text_features.float()
            logits = classifier(text_features).squeeze(1)  # [B*4]
            logits = logits.view(batch_size, 4)  # [B, 4]


            # Convert labels from one-hot to index if needed
            correct_indices = correct_ids.to(DEVICE)
            targets = torch.zeros_like(logits)
            targets.scatter_(1, correct_indices.unsqueeze(1), 1)

            loss = criterion(logits, targets)
            epoch_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == correct_indices).sum().item()
            total_samples += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = correct_total / total_samples
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {acc:.4f}")

    torch.save(classifier.state_dict(), "classifier.pth")
    print("Model saved as classifier.pth")

if __name__ == "__main__":
    main()
    

