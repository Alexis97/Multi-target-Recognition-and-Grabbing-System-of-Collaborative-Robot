from ctypes import *
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

# Load DLLs
ControlLayerDll = WinDLL(os.path.join(curPath, 'ControlLayerDll.dll')) 
libArmObj = WinDLL(os.path.join(curPath, 'libArmObj.dll')) 

# Define C struct
class AngleRange(Structure):
	_fields_ = [('lowBound', c_double),
				('highBound', c_double)]

class HSArm(Structure):
	_fields_ = [('heights', c_double * 5),
				('distances', c_double * 5),
				('ArLost_ErrData', AngleRange),
				()]