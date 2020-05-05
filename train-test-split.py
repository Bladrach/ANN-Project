import os
import glob
import shutil

data_path = glob.glob("C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset\\pl_tomato_dataset\\*") 
save_path = "C:\\Users\\Mehmet\\Desktop\\ANN\\img_dataset" 

for i in range(len(data_path)): 
    folderName = data_path[i]
    fn = str(folderName.split("\\")[7])
    os.mkdir(save_path + "\\pl_test\\" + fn)
    os.mkdir(save_path + "\\pl_train\\" + fn)
    imageFolderName = glob.glob(folderName + "/*.jpg")
    print("{}. klasör için işlem başladı. Lütfen bekleyiniz...".format(str(i + 1)))
    for j in range(len(imageFolderName)):
        if(j <= len(imageFolderName)*0.2):  # Klasörde bulunan resimlerin %20 sini test, %80 ini train olarak kaydediyor.
            shutil.copy(imageFolderName[j], save_path + "\\pl_test\\" + fn)
        else:
            shutil.copy(imageFolderName[j], save_path + "\\pl_train\\" + fn)
print("İşlem başarıyla tamamlandı!")