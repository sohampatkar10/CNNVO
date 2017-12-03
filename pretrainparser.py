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

        # Generate transform
        rx = np.random.random()
        ry = np.random.random()
        rt = np.random.random()

        dx = int(6.0*rx - 3.0)
        dy = int(6.0*ry - 3.0)
        th = 60.0*rt - 30.0

        tf = transform.SimilarityTransform(scale=1, rotation=0.0, translation=(dx, dy))

        # Transform image
        im2 = transform.warp(im1, tf)
        im2 = transform.rotate(im2, th)

        #  Bin transform into classes
        NUM_XY_CLASSES = 7
        NUM_TH_CLASSES = 20

        for xy in range(NUM_XY_CLASSES):
            if (xy - 0.5 <= rx*6.0 < xy + 0.5):
                lx = xy
            if (xy - 0.5 <= ry*6.0 < xy + 0.5):
                ly = xy
               
        for t in range(NUM_TH_CLASSES):
            if (t*3.0 - 1.5 <= rt*60.0 < t*3.0 + 1.5):
                lt = t

        dict = {"img1": im1, "img2": im2, "tf":(lx, ly, lt)}

        return dict
