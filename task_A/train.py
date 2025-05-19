import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from tqdm import tqdm
from utils import HallucinationDataset, Classifier

# Configuration
JSON_PATH = '../dataset/Hallucination Detection/json/train.json'
IMAGE_FOLDER = '../dataset/Hallucination Detection/images'
MODEL_PATH = "classifier.pth"
BATCH_SIZE = 100
EPOCHS = 10
LR = 5e-3
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
    classifier = Classifier(model.text_projection.shape[1]*3, 128, 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)

    # model.eval()  # Keep CLIP frozen unless you're fine-tuning
    print("Start training")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        correct_total = 0
        total_samples = 0

        for images, tokenized_choices, tokenized_answers, labels, correct_ids in tqdm(loader):
            images = images.to(DEVICE) # [B, 3, 224, 224]
            
            # Reshape text input: [B, 4, 77] → [B*4, 77]
            tokenized_choices = tokenized_choices.view(-1, tokenized_choices.size(-1)).to(DEVICE) # [B*4, 77] 
            tokenized_answers = tokenized_answers.view(-1, tokenized_answers.size(-1)).to(DEVICE) # [B, 77]
            

            labels = labels.to(DEVICE)
            correct_ids = correct_ids.to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(images)  # [B, 512]
                answer_features = model.encode_text(tokenized_answers)  # [B, 512]
                text_features = model.encode_text(tokenized_choices)  # [B*4, 512]

            # [TY]: combine text and image features
            image_features = image_features.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 512]
            answer_features = answer_features.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 512]
            text_features = text_features.view(BATCH_SIZE, 4, -1)            # [B, 4, 512]

            image_features = image_features.float()
            answer_features = answer_features.float()
            text_features = text_features.float()

            combined = torch.cat([image_features, text_features, answer_features], dim=-1)
            logits = classifier(combined).squeeze(-1) # [B, 4]

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

    torch.save(classifier.state_dict(), MODEL_PATH)
    print("Model saved", MODEL_PATH)

if __name__ == "__main__":
    main()
    

