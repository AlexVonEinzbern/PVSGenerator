import matplotlib.pyplot as plt
import numpy as np
from random import randint
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

class PVSCreator():

    def __init__(self):
        
        self.x = 0
        self.y = 0
        self.z = 0
        self.x_vals = 0
        self.y_vals = 0
        self.z_vals = 0
        self.voxelarray = 0
        self.colors = 0
        self.ax = 0
        self.interpolant = 0
        self.sampling_points = 0
        self.pvs_rotated = 0

    def createPVS(self, NVox):
        
        # New -- from PSVDRO --
        half_size = np.floor(NVox / 2)

        self.x_vals = np.arange(-half_size[1], half_size[1])
        self.y_vals = np.arange(-half_size[0], half_size[0])
        self.z_vals = np.arange(-half_size[2], half_size[2])

        # Prepare some coordenates
        self.x, self.y, self.z = np.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')

        # Params for cuboids
        size1 = randint(2, 3)  # think of this as the cuboid size1
        size2 = randint(2, 3)  # think of this as the cuboid size2
        size3 = randint(2, 3)  # think of this as the cuboid size3
        size4 = randint(2, 3)  # think of this as the cuboid size4

        # Create 4 cuboids
        cube1 = ((self.x > size1) & (self.x <= 2*size1) & (self.y > size1) & (self.y <= 2*size1) & (self.z <= size1))
        cube2 = ((self.x > size2) & (self.x <= 2*size2) & (self.y > size2) & (self.y <= 2*size2) & (self.z >= size1) & 
                 (self.z <= size1+size2))
        cube3 = ((self.x > size3) & (self.x <= 2*size3) & (self.y > size3) & (self.y <= 2*size3) & (self.z >= size1+size2) & 
                 (self.z <= size1+size2+size3))
        cube4 = ((self.x > size4) & (self.x <= 2*size4) & (self.y > size4) & (self.y <= 2*size4) & (self.z >= size1+size2+size3) & 
                 (self.z <= size1+size2+size3+size4))

        # # Combine the objects into a single boolean array
        self.voxelarray = cube1 | cube2 | cube3 | cube4

        # Convert self.voxelarray to a float array
        self.voxelarray = self.voxelarray.astype(float)

        self.interpolant = RegularGridInterpolator((self.x_vals, self.y_vals, self.z_vals), self.voxelarray, 
                                                   method='linear', bounds_error=False, fill_value=0)

        self.sampling_points = np.column_stack([self.x.reshape(-1,1), self.y.reshape(-1,1), self.z.reshape(-1,1)])

    def rotatePVS(self):
        # Method that returns the PVS rotated

        angles = np.random.randint(low=-20, high=20, size=3)

        # create a rotation object that rotates around the x-, y- and z-axis
        r_x = Rotation.from_euler('x', angles[0], degrees=True)
        r_y = Rotation.from_euler('y', angles[1], degrees=True)
        r_z = Rotation.from_euler('z', angles[2], degrees=True)

        # combined
        r = r_z * r_y * r_x
        rotm = r.as_matrix()

        sampling_points_rotated = np.dot(rotm, self.sampling_points.T)
        pvs_rotated = self.interpolant(sampling_points_rotated.T)
        pvs_rotated = pvs_rotated >= 0.5

        self.pvs_rotated = pvs_rotated.reshape(self.x.shape)

    def plotPVS(self):
        self.ax = plt.figure().add_subplot(projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.voxels(self.voxelarray, edgecolor='k')
        plt.show()

    # Return a numpy array
    def numpy_array(self):
        return np.float32(self.pvs_rotated)
    
class PVSAdder():
    
    def __init__(self):
        self.atlas = 0
        self.pvs = 0    

    def addPVS(self, atlas, result_dir, n, p):
        img_nii = nib.load(atlas)
        img = img_nii.get_fdata()

        # PVS_rois = [2, 10, 11, 12, 13, 16, 17, 18, 26, 41, 49, 50, 51, 52, 53, 54, 58]
        indices = np.argwhere((img == 2) | (img == 10) | (img == 11) | (img == 12) | (img == 13)
                      | (img == 16) | (img == 17) | (img == 18) | (img == 26) | (img == 41)
                      | (img == 49) | (img == 50) | (img == 51) | (img == 52) | (img == 53)
                      | (img == 54) | (img == 58))

        # Number of pvs between 20 and 100
        num_pvs = np.random.randint(low=20, high=100)
        
        for _ in range(num_pvs):

            Nvox = np.array([16, 16, 16])

            # Call the PVS generator
            pvsC = PVSCreator()
            pvsC.createPVS(Nvox)
            pvsC.rotatePVS()
            pvs = pvsC.numpy_array()
            
            # Choice a random region and add a PVS
            random_index = np.random.choice(len(indices))
            i, j, k = indices[random_index]
            img[i:i+16, j:j+16, k:k+16] = 69*(pvs>0) + np.multiply(img[i:i+16, j:j+16, k:k+16], pvs==0)

        # Create the Nifti image
        affine = img_nii.affine                                                # Same affine transform
        header = img_nii.header                                                # Same header

        img_nifti = nib.Nifti1Image(img, affine, header)
        nib.save(img_nifti, result_dir + 'image{}{}.nii.gz'.format(n, p))     # Create a file with same name and pvs_ as prefix
