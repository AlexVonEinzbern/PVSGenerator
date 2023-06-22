import os
import sys
import glob
from pvsCreator import PVSAdder

"""
Parameters:
    - num_imgs: number of images to generate
    - training_dir: directory where the images are stored
    - result_dir: directory where the images will be stored

Returns:
    - num_imgs images with the PVS added to the ROIs

Example:
    python main.py 1000 training_segs/ ~/pvs_labels/
"""

num_imgs = int(sys.argv[1])
training_dir = sys.argv[2]
result_dir = sys.argv[3]

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

file_list = glob.glob(os.path.join(training_dir, "*"))

for i in range(num_imgs):
    file = file_list[i % len(file_list)]
    mask = PVSAdder()
    mask.addPVS(file, result_dir, i)