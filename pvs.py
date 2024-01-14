from pvsCreator import PVSCreator
import numpy as np

NVox = np.array([16, 16, 16])

pvs = PVSCreator()
pvs.createPVS(NVox)
# pvs.numpy_array()
pvs.plotPVS()
pvs.rotatePVS()
pvs.plotPVS2()