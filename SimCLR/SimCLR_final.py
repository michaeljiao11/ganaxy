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
import random

from loss_function import ContrastiveLoss
from premodel import Identity, LinearLayer, ProjectionHead, PreModel
from premodel_dataset import GalaxyDataGen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PreModel("resnet50").to(device)
galmodel = GalaxySimCLR(model, num_classes).to(device)

model.load_state_dict(torch.load("PATH")["model_state_dict"])
galmodel.load_state_dict(torch.load("PATH")["model_state_dict"])
