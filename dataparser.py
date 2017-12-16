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

        self.dx = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dz = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dth = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.x = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.z = np.zeros(self.poses.shape[0]-1, dtype=np.float)

        for j in range(self.poses.shape[0]-1):
            gt = self.poses[j].reshape(4,4)
            x = gt[0,3]
            z = gt[2,3]
            self.x[j] = x
            self.z[j] = z
            gt_1 = self.poses[j+1].reshape(4,4)

            dg  = np.dot(np.linalg.inv(gt), gt_1)
            dx = dg[0,3]
            dz = dg[2,3]
            dth = G.G_to_rpy(dg)[1]
            self.dx[j] = (dx+0.1)/0.2
            self.dz[j] = dz/1.5
            self.dth[j] = (dth+0.06)/0.12

        self.dx = np.clip(self.dx, 0.0, 1.0)
        self.dz = np.clip(self.dz, 0.0, 1.0)
        self.dth = np.clip(self.dth, 0.0, 1.0)

        self.dx = torch.from_numpy(self.dx).view(self.dx.shape[0],-1).type(torch.FloatTensor).cuda()
        self.dz = torch.from_numpy(self.dz).view(self.dx.shape[0],-1).type(torch.FloatTensor).cuda()
        self.dth = torch.from_numpy(self.dth).view(self.dx.shape[0],-1).type(torch.FloatTensor).cuda()

        self.im_np = np.zeros((self.times.shape[0],3,128,128))
        for idx in range(len(self.filenames_im3)):
            img_l1 = cv2.imread(self.filenames_im3[idx])
            img_l1 = cv2.resize(img_l1, (128, 128))
            img_l1 = np.transpose(img_l1, (2,0,1))
            self.im_np[idx] = img_l1

        self.im_np = (self.im_np - np.mean(self.im_np))/np.std(self.im_np)
        self.im_np = torch.from_numpy(self.im_np).type(torch.FloatTensor).cuda()

    def __len__(self):
        return self.times.shape[0]-1

    def __getitem__(self,idx):

        img_l1 = self.im_np[idx]
        img_l2 = self.im_np[idx+1]
        dx = self.dx[idx]
        dz = self.dz[idx]
        dth = self.dth[idx]
        dt = self.times[idx+1] - self.times[idx]
        time = self.times[idx]
        x = self.x[idx]
        z = self.z[idx]
        dt = self.times[idx+1] - self.times[idx]

        data = {"img_l1": img_l1, "img_l2": img_l2, "x":x, "z":z, "dx":dx,  "dz": dz, "dth": dth, "dt": dt, "t": time}

        return data
