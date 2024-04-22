from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor

class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_dir, labels_dir):
        self.file_paths = glob.glob(os.path.join(root_dir, '*.pckl'))
        # Load the data from the text file
        data = np.genfromtxt(labels_dir, dtype='str', delimiter=' ')
        # Extract the last column which contains the labels
        labels = data[:, -1].tolist()
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sample_rate = sf.read(self.file_paths[idx])
        label = self.labels[idx]
        return audio, label

def collate_fn(batch):
    audios, labels = zip(*batch)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
    audios = feature_extractor(audios, sampling_rate=16000, return_tensors='pt', padding=True).input_values
    labels = torch.tensor(labels)
    return audios, labels
