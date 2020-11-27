import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation


class RBTUtils:
    """
    Utilities rigid body rotations and transformations
    """

    @staticmethod
    def extractRPY(G):
        return Rotation.from_matrix(G[:3, :3]).as_euler('xyz', )


    @staticmethod
    def rpyToRBT(rpy):
        return Rotation.from_euler('zyx', rpy)


    @staticmethod
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

if __name__=="__main__":
    # Just checking if imports work correctly
    pass

