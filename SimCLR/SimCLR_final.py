import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import pickle
from torch.optim.optimizer import Optimizer, required
import re
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import RMSprop
import os
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt

from loss_function import ContrastiveLoss
from premodel import Identity, LinearLayer, ProjectionHead, PreModel
from premodel_dataset import GalaxyDataGen
from simclr_classifier_dataset import SUPERGALAXYDataGen
from simclr_classifier_model import GalaxySimCLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PreModel("resnet50").to(device)
model.load_state_dict(torch.load("PATH")["model_state_dict"])

galmodel = GalaxySimCLR(model, num_classes).to(device)
galmodel.load_state_dict(torch.load("PATH")["model_state_dict"])


data_folder = "PATH"
ds = SUPERGALAXYDataGen(data_folder, 'valid', num_classes=2)
dl = torch.utils.data.DataLoader(overall_ds, batch_size=32, shuffle=True, num_workers=2, drop_last = True)
