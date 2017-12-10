import torch
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import transform, util
import G_to_rpy as G

class DataParser(Dataset):
    '''
    Class for parsing data
    Args:
    root_dir: root directory of file
    sequence: sequence from which data to load
    '''

    def __init__(self, sequence, root_dir="../kitti_datasets"):
        print "loading data from sequence", sequence
        # Right images
        self.filenames_im2 = [glob.glob(root_dir+'/sequences/'+sequence+'/image_2/*.png')]
        self.filenames_im2 = self.filenames_im2[0]
        self.filenames_im2.sort()

        # Left images
        self.filenames_im3 = [glob.glob(root_dir+'/sequences/'+sequence+'/image_3/*.png')]
        self.filenames_im3 = self.filenames_im3[0]
        self.filenames_im3.sort()

        # Timestamps
        filename_times = root_dir + '/sequences/'+sequence + '/times.txt'
        time_data = open(filename_times,'r')
        time = time_data.readlines()
        self.times = np.zeros((len(time),1), dtype=np.float)
        for i in range(len(time)):
            self.times[i] = time[i].split()

        # Poses
        filename_poses = root_dir + '/poses/'+sequence + '.txt'
        pose_data = open(filename_poses, 'r')
        pose_str = pose_data.readlines()
        self.poses = np.zeros((len(pose_str),16), dtype=np.float)
        for i in range(len(pose_str)):
            self.poses[i][0:12] = pose_str[i].split()
            self.poses[i][12:16] = [0.0, 0.0, 0.0, 1.0]

    def __len__(self):
        return self.times.shape[0]-1

    def __getitem__(self,idx):

        # frame at timestep t 
        img_r1 = cv2.imread(self.filenames_im2[idx])
        img_r1 = cv2.resize(img_r1, (128, 128))
	img_r1 = np.transpose(img_r1, (2,0,1))
	img_r1 = img_r1/(np.max(img_r1)-np.min(img_r1))

        img_l1 = cv2.imread(self.filenames_im3[idx])
        img_l1 = cv2.resize(img_l1, (128, 128))
	img_l1 = np.transpose(img_l1, (2,0,1))
	img_l1 = img_l1/(np.max(img_l1)-np.min(img_l1))

        # frame at timestep t+1
        img_r2 = cv2.imread(self.filenames_im2[idx+1])
        img_r2 = cv2.resize(img_r2, (128, 128))
	img_r2 = np.transpose(img_r2, (2,0,1))
	img_r2 = img_r2/(np.max(img_r2)-np.min(img_r2))

        img_l2 = cv2.imread(self.filenames_im3[idx+1])
        img_l2 = cv2.resize(img_l2, (128, 128))
	img_l2 = np.transpose(img_l2, (2,0,1))
	img_l2 = img_l2/(np.max(img_l2)-np.min(img_l2))

        gt = self.poses[idx].reshape(4,4)
        gt_1 = self.poses[idx+1].reshape(4,4)

        dR  = np.dot(np.linalg.inv(gt[:3, :3]), gt_1[:3,:3])
        dx = gt[0][3] - gt_1[0][3]
        dz = gt[2][3] - gt_1[2][3]
        th = G.G_to_rpy(dR)[2]*180/3.14

        #  Bin transform into classes
        NUM_XY_CLASSES = 21
        NUM_TH_CLASSES = 21

        for xy in range(NUM_XY_CLASSES):
            if (6.0/20.0*(xy - 0.5) <= dx + 3.0 < 6.0/20.0*(xy + 0.5)):
                lx = xy
            if (6.0/20.0*(xy - 0.5) <= dz + 3.0 < 6.0/20.0*(xy + 0.5)):
                lz = xy

        for t in range(NUM_TH_CLASSES):
            if (2.4/20.0*(t - 0.5) <= th + 1.2 < 2.4/20.0*(t + 0.5)):
                lt = t

        dt = self.times[idx+1] - self.times[idx]
	
        data = {"img_l1": img_l1, "img_r1": img_r1, "img_l2": img_l2, "img_r2": img_r2, "dx":lx,  "dz": lz, "dth": lt, "dt": dt}

        return data
