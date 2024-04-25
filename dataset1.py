from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor

class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_dir, labels_dir):
        #self.file_paths = glob.glob(os.path.join(root_dir, '*.pckl'))
        # Load the data from the text file
        data = np.genfromtxt(labels_dir, dtype='str', delimiter=' ')
        # Extract the last column which contains the labels
        labels = data[:, -1].tolist()
        files= data[:, 1].tolist()
        self.file_paths = [root_dir + file +'.flac' for file in files]
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
    # Convert labels from 'bonafide'/'spoof' to 0/1 integers if they are not already integers
    labels = [0 if label == 'spoof' else 1 for label in labels]
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are torch.long for classification
    return {"input_values": audios, "labels": labels}



# traindata=ASVspoof2019Dataset('D:/dataset/LA/LA/ASVspoof2019_LA_train/flac/','D:/dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
# t=1