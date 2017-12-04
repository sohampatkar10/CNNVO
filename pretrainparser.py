import torch
import torchvision 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import transform
import cv2

class PreTrainParser():
    def __init__(self, train = True):
        self.cifar = torchvision.datasets.MNIST('../',download=True, train=train)
        
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        im1 = np.array(self.cifar[idx][0])
        # Generate transform
        rx = np.random.random()
        ry = np.random.random()
        rt = np.random.random()

        dx = int(6.0*rx - 3.0)
        dy = int(6.0*ry - 3.0)
        th = 60.0*rt - 30.0

        M1 = cv2.getRotationMatrix2D((14,14), th, 1)
        im2= cv2.warpAffine(im1, M1, (28, 28))
        M2 = np.float32([[1,0, dx],[0,1,dy]])
        im2 = cv2.warpAffine(im2,M2,(28,28))

        #  Bin transform into classes
        NUM_XY_CLASSES = 7
        NUM_TH_CLASSES = 21

        for xy in range(NUM_XY_CLASSES):
            if (xy - 0.5 <= rx*6.0 < xy + 0.5):
                lx = xy
            if (xy - 0.5 <= ry*6.0 < xy + 0.5):
                ly = xy
               
        for t in range(NUM_TH_CLASSES):
            if (t*3.0 - 1.5 <= rt*60.0 < t*3.0 + 1.5):
                lt = t

        im1 = im1/(np.max(im1)-np.min(im1))
        im2 = im2/(np.max(im2)-np.min(im2))
        dict = {"img1": im1, "img2": im2, "tf":np.array([lx, ly, lt])}
        #dict = {"img1": im1, "img2": im2, "tf":np.array([2, 3, 15])}

        return dict
