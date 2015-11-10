from __future__ import division
import numpy as np
import scipy

def convert2db(x):
	return 20 * np.log10(x)

def normalize(x):
	x = x/(np.amax(np.absolute(x)))