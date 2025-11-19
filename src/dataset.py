import os
import torch
import torchaudio
from torch.utils.data import Dataset

class GTZANDataset(Dataset):
    def __init__(self, root_dir, sample_rate=22050, n_mels=64, duration=3):
        self.root_dir = root_dir
        self.sample_rate = sample_rate #Sample Rate (Hz) for audio. 22050 is common.
        self.n_mels = n_mels
        self.duration = duration
        self.max_len = sample_rate * duration #Fixed 3 second clip, adjust if needed

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels
        )

        self.file_paths = []
        self.labels = []
        self.genres = sorted(os.listdir(root_dir))

        for label, genre in enumerate(self.genres):
            genre_dir = os.path.join(root_dir, genre)
            for file in os.listdir(genre_dir):
                if file.endswith(".wav"):
                    self.file_paths.append(genre_dir, file)
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path) #waveform = raw audio as a tensor [channels, samples]

        #Resampling
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample)

        #Fix audio length (pad or cut)
        if waveform.shape[1] < self.max_len():
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.max_len]

        mel = self.mel_transform(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        #Add channel dim
        return mel.unsqueeze(0), label 
        