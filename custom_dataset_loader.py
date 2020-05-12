from torch.utils.data import Dataset
import glob
import cv2
from autoCanny import auto_canny


# intel için denendi
class MyCustomDataset(Dataset):
    def __init__(self, data_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "\\edg_train\\*")
        else:
            self.data_path = glob.glob(data_path + "\\edg_test\\*")

    def __getitem__(self, index):
        
        folderName = self.data_path[index]
        fn = int(folderName.split("\\")[8].split("_")[0])
        imageFolderName = glob.glob(folderName + "/*.png")
        img = cv2.imread(str(imageFolderName[index]), 0)   # Grayscale okumak için 0 koyuldu.
        #print(type(img))
        #print(fn)
        return img, fn


    def __len__(self):
        return len(self.data_path)


class MyCustomFruitDataset(Dataset):
    def __init__(self, data_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "\\Training\\*")
        else:
            self.data_path = glob.glob(data_path + "\\Test\\*")

    def __getitem__(self, index):
        
        folderName = self.data_path[index]
        fn = int(folderName.split("\\")[9].split("_")[0])
        imageFolderName = glob.glob(folderName + "/*.jpg")
        img = cv2.imread(str(imageFolderName[index]))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        # perform the canny edge detector to detect image edges
        img = auto_canny(blurred, sigma = 0.33)

        # resize to 50x50
        dim = (50, 50)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        #log_spec = torch.from_numpy(log_spec)
        #print(type(img))
        #print(fn)
        return img, fn


    def __len__(self):
        return len(self.data_path)
