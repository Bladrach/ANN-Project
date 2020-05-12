import os
import glob
import shutil
import numpy as np
import cv2
from autoCanny import auto_canny

data_path = glob.glob("C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\intel-image-classification\\seg_train\\*") 
save_path = "C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\edged_intel" 

for i in range(len(data_path)): 
    folderName = data_path[i]
    fn = str(folderName.split("\\")[8])
    os.mkdir(save_path + "\\edg_train\\" + fn)
    imageFolderName = glob.glob(folderName + "/*.jpg")
    print("{}. klasör için işlem başladı. Lütfen bekleyiniz...".format(str(i + 1)))
    for j in range(len(imageFolderName)):
        #reading the image 
        image = cv2.imread(str(imageFolderName[j]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        # perform the canny edge detector to detect image edges
        edged_canny = auto_canny(blurred, sigma = 0.33)
        imageName = str(imageFolderName[j].split("\\")[9].split('.')[0])
        cv2.imwrite(save_path + "\\edg_train\\" + fn + '\\{}.png'.format(imageName), edged_canny)

print("İşlem başarıyla tamamlandı!")