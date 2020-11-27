import cv2
import glob
import numpy as np
from skimage import transform, util
import RBTUtils as RBU

class DataParser():
    '''
    Class for parsing data
    Args:
    root_dir: root directory of file
    sequence: sequence from which data to load
    '''

    # RRR: This takes only left image for now, we want all the data.
    #      Not sure of the format yet so will defer this to after that
    @staticmethod
    def getImages(root_dir, sequence):
        filenames = [
            glob.glob(root_dir + '/sequences/' + sequence + '/image_3/*.png')]
        filenames = self.filenames[0]
        filenames.sort()
        return filenames


    @staticmethod
    def getTimestamps(root_dir, sequence):
        filename_times = root_dir + '/sequences/' + sequence + '/times.txt'
        time_data = open(filename_times, 'r')
        time = time_data.readlines()
        times = np.zeros((len(time), 1), dtype=np.float)

        # RRR: Can pythonify this
        for i in range(len(time)):
            times[i] = time[i].split()

        return times


    @staticmethod
    def getPoses(root_dir, sequence):
        filename_poses = root_dir + '/poses/'+sequence + '.txt'
        pose_data = open(filename_poses, 'r')
        pose_str = pose_data.readlines()
        poses = np.zeros((len(pose_str), 16), dtype=np.float)

        # RRR: Can pythonify this
        for i in range(len(pose_str)):
            poses[i][0:12] = pose_str[i].split()
            poses[i][12:16] = [0.0, 0.0, 0.0, 1.0]

        return poses


    def __init__(self, root_dir, sequence):
        print "loading data from sequence", sequence

        # Left images
        self.image_filenames = DataParser.getImages(root_dir, sequence)

        # Timestamps
        self.times = DataParser.getTimestamps(root_dir, sequence);

        # Poses
        self.poses = DataParser.getPoses(root_dir, sequence)

        # RRR: Sticking to planar for now, because unsure of how to handle
        #      rotations
        self.dx  = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dz  = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.dth = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.x   = np.zeros(self.poses.shape[0]-1, dtype=np.float)
        self.z   = np.zeros(self.poses.shape[0]-1, dtype=np.float)

        for j in range(self.poses.shape[0]-1):
            gt = self.poses[j].reshape(4, 4)
            (self.x[j], self.z[j]) = (gt[0,3], gt[2,3])
            gt_1 = self.poses[j+1].reshape(4, 4)

            # RRR: Confirm this math
            # dg = G(t).inverse() * G(t-1)
            dg = np.dot(np.linalg.inv(gt), gt_1)
            (self.dx[j], self.dy[j], self.dz[j]) = dg[:,3]
            dth = RBU.extractRPY(dg)[1]

        # RRR: Could paramify the size
        self.images = np.zeros((self.times.shape[0], 3, 128, 128))
        for idx in range(len(self.image_filenames)):
            img_l1 = cv2.imread(self.image_filenames[idx])
            img_l1 = cv2.resize(img_l1, (128, 128))
            img_l1 = np.transpose(img_l1, (2, 0, 1))
            self.images[idx] = img_l1


    def __len__(self):
        return self.times.shape[0]-1


    def __getitem__(self, idx):
        data = {
                 "img_l1": self.images[idx],
                 "img_l2": self.images[idx+1],
                 "x": self.x[idx],
                 "z": self.z[idx],
                 "dx": self.dx[idx],
                 "dz": self.dz[idx],
                 "dth": self.dth[idx],
                 "dt": self.times[idx+1] - self.times[idx],
                 "t": self.times[idx]
                }

        return data


    if __name__=="__main__":
        # Just checking if imports work correctly
        pass
