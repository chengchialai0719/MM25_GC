import os
import json
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
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(json_path, image_folder, model_path, output_path="results.json"):
    # Load CLIP model and classifier
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    classifier = Classifier(clip_model.text_projection.shape[1]*3, 128, 1).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
    classifier.eval()

    # Dataset
    tokenizer = clip.tokenize
    dataset = HallucinationDataset(json_path, image_folder, tokenizer, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct_total = 0
    total_samples = 0

    # [TY]: compute F1 score
    metric = BinaryF1Score().to(DEVICE)
    results = []
    with torch.no_grad():
        for images, tokenized_choices, tokenized_answers, labels, correct_ids in tqdm(loader):
            # BATCH_SIZE = images.size(0)
            
            images = images.to(DEVICE)
            tokenized_choices = tokenized_choices.view(-1, tokenized_choices.size(-1)).to(DEVICE)
            tokenized_answers = tokenized_answers.view(-1, tokenized_answers.size(-1)).to(DEVICE) # [B, 77]
            labels = labels.to(DEVICE)
            correct_ids = correct_ids.to(DEVICE)

            # ~~Extract text features~~
            # [TY]: combine text and image features
            image_features = clip_model.encode_image(images)  # [B, 512]
            answer_features = clip_model.encode_text(tokenized_answers)  # [B, 512]
            text_features = clip_model.encode_text(tokenized_choices)  # [B*4, 512]

            image_features = image_features.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 512]
            answer_features = answer_features.unsqueeze(1).repeat(1, 4, 1)  # [B, 4, 512]
            text_features = text_features.view(BATCH_SIZE, 4, -1)            # [B, 4, 512]

            image_features = image_features.float()
            answer_features = answer_features.float()
            text_features = text_features.float()

            combined = torch.cat([image_features, text_features, answer_features], dim=-1)
            logits = classifier(combined).squeeze(-1) # [B, 4]

            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == correct_ids).sum().item()
            total_samples += labels.size(0)
            # [TY]: compute F1 score
            metric.update(logits, labels)

            # Save results
            for i in range(BATCH_SIZE):
                results.append({
                    "image_id": dataset.data[i]["image_id"],
                    "predicted_choice": chr(predictions[i].item() + ord("A")),
                    "correct_choice": dataset.data[i]["correct_choice"],
                })

    acc = correct_total / total_samples
    # [TY]: compute F1 score
    f1 = metric.compute()
    print(f"Evaluation Accuracy on {json_path}: acc {acc:.4f}, F1 {f1:.4f}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to val/test JSON file", default="../dataset/Hallucination Detection/json/test.json")
    parser.add_argument("--image_folder", type=str, default="../dataset/Hallucination Detection/images")
    parser.add_argument("--model_path", type=str, default="classifier.pth")
    args = parser.parse_args()

    evaluate(args.json, args.image_folder, args.model_path)
