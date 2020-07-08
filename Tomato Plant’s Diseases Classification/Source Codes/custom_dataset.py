from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import cv2
from sklearn.preprocessing import MinMaxScaler


class CustomDataset(Dataset):
    def __init__(self, data_path, train = False, val = False):
        if (train == True):
            self.data_path = glob.glob(data_path + "\\pv\\train\\*.jpg")
        elif (val == True):
            self.data_path = glob.glob(data_path + "\\pv\\val\\*.jpg")
        else:
            self.data_path = glob.glob(data_path + "\\pv\\test\\*.jpg")
        
    def __getitem__(self, index):
        
        imgName = self.data_path[index]
        image = cv2.imread(imgName, -1)
        label = int(imgName.split("\\")[7].split("_")[0])

        dim = (100, 100)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # feature-descriptor-1: Hu Moments
        feature = cv2.HuMoments(cv2.moments(gray)).flatten()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # compute the color histogram
        bins = 8
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        global_features = np.hstack([hist, feature])
        global_features = np.reshape(global_features, (-1,1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_global_features = scaler.fit_transform(global_features).astype(np.float32)

        return normalized_global_features, label

    def __len__(self):
        return len(self.data_path)
