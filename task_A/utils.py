###
# Model Architecture
# Input: image + choice * 4
# Text tokenizer: clip
# Image tokenizer: clip
# Classifier: 2 FC + 1 ReLU
###

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import clip  # Needs: pip install git+https://github.com/openai/CLIP.git

# [TY]: 2 FC + 1 ReLU layers
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.relu = nn.ReLU()                         # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second linear layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
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
        answer = sample['system1_answer']
        choices = sample["choices"]

        texts = []
        for choice in choices:
            # text = f"Question: {question} Description: {description} Choice: {choice['choice']}"
            text = choice['choice']
            texts.append(text)
            
        MAX_TOKENS = 77
        tokenized_choice = [self.tokenizer(t, truncate=True)[0] for t in texts]
        tokenized_choice = torch.stack(tokenized_choice)  # shape: [4, 77]

        tokenized_text = self.tokenizer(answer, truncate=True)

        # encoded_texts = [self.tokenizer(text)]
        correct_choice_id = ord(sample["correct_choice"]) - ord("A")
        labels = torch.zeros(len(choices))
        labels[correct_choice_id] = 1

        
        return image_tensor, tokenized_choice, tokenized_text, labels, correct_choice_id
