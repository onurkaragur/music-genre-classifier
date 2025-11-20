import os
import torch
import torchaudio
import librosa
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
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )

        self.file_paths = []
        self.labels = []
        self.genres = sorted(os.listdir(root_dir))

        # Filter out corrupted files during initialization
        print(f"Loading dataset from {root_dir}...")
        for label, genre in enumerate(self.genres):
            genre_dir = os.path.join(root_dir, genre)
            for file in os.listdir(genre_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_dir, file)
                    # Quick check if file can be loaded
                    try:
                        librosa.load(file_path, sr=None, duration=0.1, mono=True)
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    except Exception:
                        print(f"Warning: Skipping corrupted file {file_path}")
        print(f"Loaded {len(self.file_paths)} valid audio files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio using librosa (more robust for various formats)
        try:
            data, sr = librosa.load(path, sr=None, mono=False)
        except Exception:
            # Fallback: try loading as mono
            data, sr = librosa.load(path, sr=None, mono=True)
        
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)  # Convert from [samples, channels] to [channels, samples]

        #Resampling
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        #Fix audio length (pad or cut)
        if waveform.shape[1] < self.max_len:
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.max_len]

        mel = self.mel_transform(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # mel shape is [channels, n_mels, time_frames]
        # We need [1, n_mels, time_frames] for Conv2d which expects [batch, channels, height, width]
        if mel.dim() == 3:
            # If stereo, take first channel or average
            mel = mel[0] if mel.shape[0] == 2 else mel.squeeze(0)
        # Add channel dimension: [n_mels, time_frames] -> [1, n_mels, time_frames]
        mel = mel.unsqueeze(0)
        
        return mel, label 
        