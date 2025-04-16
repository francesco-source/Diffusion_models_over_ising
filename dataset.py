import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

class CustomDataset(Dataset):
    def __init__(self, data_path, categories, num_classes):
        self.data = []
        for category in categories:
            category_path = os.path.join(data_path, category)
            for filename in tqdm(os.listdir(category_path)):
                path = os.path.join(category_path, filename)
                img = np.load(path).astype(np.float32)  # Load and convert image
                _, temp_label = self.one_hot_label(filename)
                temp_label = torch.nn.functional.one_hot(torch.tensor(temp_label), num_classes=num_classes).float()
                self.data.append((torch.tensor(img).unsqueeze(0),temp_label))  # Add channel dim
        random.shuffle(self.data)
    def one_hot_label(self, filename):
        label, T, _ = filename.split('_')
        temp_label = round((float(T) - 1.5) / 0.1)
        return torch.tensor([1, 0]) if label == 'low' else torch.tensor([0, 1]), temp_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
