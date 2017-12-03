import torch
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import transform, utils

class DataParser(Dataset):
  '''
  Class for parsing data
  Args:
    root_dir: root directory of file
    sequence: sequence from which data to load
  '''

  def __init__(self, root_dir, sequence):
    print "loading data from sequence", sequence
    # Right images
    self.filenames_im2 = [glob.glob(root_dir+'/sequences/'+sequence+'/image_2/*.png')]
    self.filenames_im2 = self.filenames_im2[0]
    self.filenames_im2.sort()

    # Left images
    filenames_im3 = [glob.glob(root_dir+'/sequences/'+sequence+'/image_3/*.png')]
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
    self.poses = poses = np.zeros((len(pose),16), dtype=np.float)
    for i in range(len(pose_str)):
      self.poses[i][0:12] = pose_str[i].split()
      self.poses[i][12:16] = [0.0, 0.0, 0.0, 1.0]

  def __len__(self):
    return self.times.shape[0]

  def __getitem__(self,idx):

    img_r = cv2.imread(self.filenames_im2[idx])
    img_r = transform.rescale(img_r, 0.2)

    img_l = cv2.imread(self.filenames_im3[idx])
    img_l = transform.rescale(img_l, 0.2)

    pose = self.poses[i].reshape(4,4)
    time = self.times[i]

    data = {"img_l" : img_l, "img_r" : img_r, "pose" : pose, "time" : time}
    return data


