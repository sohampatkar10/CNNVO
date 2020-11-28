import cv2
import glob
import numpy as np
from skimage import transform, util
from utils.RBTUtils import RBTUtils as RBU
import os

from torch.utils.data import Dataset, DataLoader


class DataParser(Dataset):
    '''
    Class for parsing image, pose and timestamps from KITTI dataset
    '''


    @staticmethod
    def getImageFilenames(root_dir, sequence):
        left_image_filenames = glob.glob(root_dir + 'images_color/sequences/' + sequence + '/image_2/*.png')
        left_image_filenames.sort()

        right_image_filenames = glob.glob(root_dir + 'images_color/sequences/' + sequence + '/image_3/*.png')
        right_image_filenames.sort()
        return left_image_filenames, right_image_filenames


    @staticmethod
    def getTimestamps(root_dir, sequence):
        filename_times = root_dir + 'images_color/sequences/' + sequence + '/times.txt'
        time_data = open(filename_times, 'r')
        time = time_data.readlines()
        times = np.zeros((len(time), 1), dtype=np.float)

        # RRR: Can pythonify this
        for i in range(len(time)):
            times[i] = time[i].split()

        return times


    @staticmethod
    def getPoses(root_dir, sequence):
        filename_poses = root_dir + 'poses/' + sequence + '.txt'
        pose_data = open(filename_poses, 'r')
        pose_str = pose_data.readlines()
        poses = np.zeros((len(pose_str), 4, 4), dtype=np.float)

        # RRR: Can pythonify this
        for i in range(len(pose_str)):
            poses[i, :3, :4] = np.array(pose_str[i].split()).reshape(3,4)
            poses[i, 3] = [0.0, 0.0, 0.0, 1.0]

        return poses

    @staticmethod
    def getImage(image_filename, transform):
        img = cv2.imread(image_filename)
        img = cv2.resize(img, (128, 128))
        if transform:
            transform(img)

        return img


    def getRelevantData(self, idx):
        print("getting data for", idx)
        gt = self.poses[idx]
        self.x[idx], self.z[idx] = (gt[0,3], gt[2,3])
        gt_1 = self.poses[idx + 1]

        # RRR: Confirm this math
        # dg = G(t).inverse() * G(t-1)
        dg = np.dot(np.linalg.inv(gt), gt_1)
        self.dx[idx], self.dz[idx] = (dg[0, 3], dg[2, 3])
        dth = RBU.extractRPY(dg)[1]

        self.left_images[idx]      = DataParser.getImage(self.left_image_filenames[idx], self.transform)
        self.left_images[idx + 1]  = DataParser.getImage(self.left_image_filenames[idx + 1], self.transform)
        self.right_images[idx]     = DataParser.getImage(self.right_image_filenames[idx], self.transform)
        self.right_images[idx + 1] = DataParser.getImage(self.right_image_filenames[idx + 1], self.transform)

        self.data[idx] = {"img_l1": self.left_images[idx],
                          "img_l2": self.left_images[idx + 1],
                          "img_r1": self.right_images[idx + 1],
                          "img_r2": self.right_images[idx + 1],
                          "x"     : self.x[idx],
                          "z"     : self.z[idx],
                          "dx"    : self.dx[idx],
                          "dz"    : self.dz[idx],
                          "dth"   : self.dth[idx],
                          "dt"    : self.times[idx+1] - self.times[idx],
                          "t"     : self.times[idx]}

    def __init__(self, sequence, transform=None):
        print ("loading data from sequence", sequence)

        ROOT_DIR = os.getenv('CNNVO_DATA_ROOT')

        # Images
        self.left_image_filenames, self.right_image_filenames = DataParser.getImageFilenames(ROOT_DIR, sequence)
        assert(len(self.left_image_filenames) == len(self.right_image_filenames))

        # Timestamps
        self.times = DataParser.getTimestamps(ROOT_DIR, sequence);
        assert(len(self.left_image_filenames) == len(self.times))

        # Poses
        self.poses = DataParser.getPoses(ROOT_DIR, sequence)
        assert(len(self.left_image_filenames) == len(self.poses))

        # RRR: Sticking to planar for now, because unsure of how to handle
        #      rotations
        self.dx  = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dz  = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dth = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.x   = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.z   = np.zeros(self.poses.shape[0]-1, dtype=np.float)


        # RRR: Could paramify the size
        self.left_images = np.zeros((self.times.shape[0], 128, 128, 3))
        self.right_images = np.zeros((self.times.shape[0], 128, 128, 3))

        self.transform = transform
        self.data = dict()

    def __len__(self):
        return self.times.shape[0] - 1


    def __getitem__(self, idx):
        if not idx in self.data:
            self.getRelevantData(idx)

        return self.data[idx]


    if __name__=="__main__":
        # Just checking if imports work correctly
        pass
