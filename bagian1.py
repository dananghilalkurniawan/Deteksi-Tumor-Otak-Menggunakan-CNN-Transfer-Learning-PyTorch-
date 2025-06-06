# Import Library dan Setup Device
import os
import shutil
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import time
import copy

import matplotlib.pyplot as plt
import numpy as np

# Menyesuaikan akan menggunakan GPU atau CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device yang digunakan: {device}")
