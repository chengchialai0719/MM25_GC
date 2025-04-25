import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from utils import HallucinationDataset
from tqdm import tqdm
import argparse

# Settings
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(json_path, image_folder, model_path):
    # Load CLIP model and classifier
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    classifier = nn.Linear(clip_model.text_projection.shape[1], 1).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
    classifier.eval()

    # Dataset
    tokenizer = clip.tokenize
    dataset = HallucinationDataset(json_path, image_folder, tokenizer, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct_total = 0
    total_samples = 0

    with torch.no_grad():
        for images, tokenized_texts, labels, correct_ids in tqdm(loader):
            images = images.to(DEVICE)
            tokenized_texts = tokenized_texts.view(-1, tokenized_texts.size(-1)).to(DEVICE)
            correct_ids = correct_ids.to(DEVICE)

            # Extract text features
            text_features = clip_model.encode_text(tokenized_texts).float()
            logits = classifier(text_features).squeeze(1)
            logits = logits.view(-1, 4)

            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == correct_ids).sum().item()
            total_samples += labels.size(0)

    acc = correct_total / total_samples
    print(f"Evaluation Accuracy on {json_path}: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to val/test JSON file", default="../dataset/Hallucination Detection/json/val.json")
    parser.add_argument("--image_folder", type=str, default="../dataset/Hallucination Detection/images")
    parser.add_argument("--model_path", type=str, default="classifier.pth")
    args = parser.parse_args()

    evaluate(args.json, args.image_folder, args.model_path)
