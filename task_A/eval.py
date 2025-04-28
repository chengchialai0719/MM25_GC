import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from torchvision import transforms
import clip
from utils import HallucinationDataset, Classifier
from tqdm import tqdm
import argparse

# Settings
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(json_path, image_folder, model_path):
    # Load CLIP model and classifier
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    classifier = Classifier(clip_model.text_projection.shape[1]*2, 128, 1).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
    classifier.eval()

    # Dataset
    tokenizer = clip.tokenize
    dataset = HallucinationDataset(json_path, image_folder, tokenizer, preprocess)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    correct_total = 0
    total_samples = 0

    # [TY]: compute F1 score
    metric = BinaryF1Score().to(DEVICE)

    with torch.no_grad():
        for images, tokenized_texts, labels, correct_ids in tqdm(loader):
            batch_size = images.shape[0]
            images = images.to(DEVICE)
            tokenized_texts = tokenized_texts.view(-1, tokenized_texts.size(-1)).to(DEVICE)
            labels = labels.to(DEVICE)
            correct_ids = correct_ids.to(DEVICE)

            # ~~Extract text features~~
            # [TY]: combine text and image features
            image_features = clip_model.encode_image(images)  # [B, 512]
            text_features = clip_model.encode_text(tokenized_texts)  # [B*4, 512]

            image_features = image_features.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 512]
            text_features = text_features.view(batch_size, 4, -1)            # [B, 4, 512]
            text_features = text_features.float()

            combined = torch.cat([image_features, text_features], dim=-1)
            logits = classifier(combined).squeeze(-1) # [B, 4]

            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == correct_ids).sum().item()
            total_samples += labels.size(0)
            # [TY]: compute F1 score
            metric.update(logits, labels)

    acc = correct_total / total_samples
    # [TY]: compute F1 score
    f1 = metric.compute()
    print(f"Evaluation Accuracy on {json_path}: acc {acc:.4f}, F1 {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to val/test JSON file", default="../dataset/Hallucination Detection/json/val.json")
    parser.add_argument("--image_folder", type=str, default="../dataset/Hallucination Detection/images")
    parser.add_argument("--model_path", type=str, default="classifier.pth")
    args = parser.parse_args()

    evaluate(args.json, args.image_folder, args.model_path)
