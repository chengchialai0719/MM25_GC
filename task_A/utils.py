import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import clip  # Needs: pip install git+https://github.com/openai/CLIP.git

class HallucinationDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, preprocess):
        self.data = json.load(open(json_path))
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_folder, sample["image_id"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        
        question = sample['system1_question']
        description = sample['system1_answer']
        choices = sample["choices"]

        texts = []
        for choice in choices:
            text = f"Question: {question} Description: {description} Choice: {choice['choice']}"
            texts.append(text)
            
        MAX_TOKENS = 77
        tokenized = [self.tokenizer(t, truncate=True)[0] for t in texts]
        tokenized = torch.stack(tokenized)  # shape: [4, 77]

        # encoded_texts = [self.tokenizer(text)]
        correct_choice_id = ord(sample["correct_choice"]) - ord("A")
        labels = torch.zeros(len(choices))
        labels[correct_choice_id] = 1

        
        return image_tensor, tokenized, labels, correct_choice_id
