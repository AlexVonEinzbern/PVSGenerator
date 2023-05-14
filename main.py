import os
import glob
from pvsCreator import PVSAdder

# specify the folder path where the files are located
folder_path = "training_segs/"
result_dir = "pvs_training_segs/"

# get a list of all the files in the folder
file_list = glob.glob(os.path.join(folder_path, "*"))

# loop through each file and apply my_method from MyClass to it
k=0
for i in range(1):
    for file in file_list:
        mask = PVSAdder()
        mask.addPVS(file, result_dir, i, k)
        k+=1
    k=0
