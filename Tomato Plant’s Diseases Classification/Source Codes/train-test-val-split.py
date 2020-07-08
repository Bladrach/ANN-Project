import os
import glob
import shutil

data_path = glob.glob("C:\\Users\\Mehmet\\Desktop\\yeniANN\\raw_datasets\\PlantVillage\\*") 
save_path = "C:\\Users\\Mehmet\\Desktop\\yeniANN\\pv" 

for i in range(len(data_path)): 
    folderName = data_path[i]
    fn = str(folderName.split("\\")[7])
    os.mkdir(save_path + "\\test\\" + fn)
    os.mkdir(save_path + "\\train\\" + fn)
    os.mkdir(save_path + "\\val\\" + fn)
    imageFolderName = glob.glob(folderName + "\\*")
    print("{}. klasör için işlem başladı. Lütfen bekleyiniz...".format(str(i + 1)))
    for j in range(len(imageFolderName)):
        if(j <= len(imageFolderName)*0.1):  # Klasörde bulunan resimlerin %10'unu test, %10'unu val ve %80'ini train olarak kaydediyoruz.
            shutil.copy(imageFolderName[j], save_path + "\\test\\" + fn)
        elif( len(imageFolderName)*0.1 < j <= len(imageFolderName)*0.2):
            shutil.copy(imageFolderName[j], save_path + "\\val\\" + fn)
        else:
            shutil.copy(imageFolderName[j], save_path + "\\train\\" + fn)
print("İşlem başarıyla tamamlandı!")
