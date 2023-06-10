from pvsCreator import PVSCreator
import numpy as np

NVox = np.array([8, 8, 8])

pvs = PVSCreator()
pvs.createPVS2(NVox)
# pvs.rotatePVS()
pvs.plotPVS()