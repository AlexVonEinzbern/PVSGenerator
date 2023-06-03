import os
import sys
import glob
from pvsCreator import PVSAdder

# specify the folder path where the files are located
folder_path = "training_segs/"
result_dir = "pvs_training_segs/"

num_imgs = int(sys.argv[1])

# get a list of all the files in the folder
file_list = glob.glob(os.path.join(folder_path, "*"))

# loop through each file and apply my_method from MyClass to it
k = 0
for i in range(num_imgs):
    file = file_list[i % len(file_list)]
    mask = PVSAdder()
    mask.addPVS(file, result_dir, k)
    k += 1