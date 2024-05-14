import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, random_split
import torchaudio
from datasets import load_dataset

matplotlib.use("TkAgg")
torch.manual_seed(42)
torch.cuda.set_device(0) # 3090

waveform, sample_rate = torchaudio.load('LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
print(waveform[0][20000:21000])
print(sample_rate)

with open('LibriSpeech/dev-clean/84/121123/84-121123.trans.txt', 'r') as f:
    transcript = f.read()
print(transcript)
