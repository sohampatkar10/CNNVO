import torch
import torchvision 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import transform

class PreTrainParser():
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST('./', train=True)
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        im1 = np.array(mnist[0][0])
        dx = int(6.0*np.random.random() - 3.0)
        dy = int(6.0*np.random.random() - 3.0)
        th = 60.0*np.random.random() - 30.0

        tf = transform.SimilarityTransform(scale=1, rotation=0.0, translation=(dx, dy))

        im2 = transform.warp(im1, tf)
        im2 = transform.rotate(im2, th)
        
        dict = {"img1": im1, "img2": im2, "tf":(dx, dy, th)}
        
        return dict