import os
from shutil import copyfile


def split(source):
    for ind, file in enumerate(os.listdir(source)):
       for index, imgname in enumerate(os.listdir(source + "\\" + file)):
           copyfile(source + "\\" + file + "\\" + imgname, source + "\\" + str(ind) + "_" + str(index) + ".jpg")


if __name__ == '__main__':
    split("C:\\Users\\Mehmet\\Desktop\\yeniANN\\pv\\val")
