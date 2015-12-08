from __future__ import division
import numpy as np
import scipy

def import_audio(filepath):
	"""
	filepath: string pointing to audio file on disk

	Simple function to quickly read audio data into numpy array
	
	returns: data = audio in samples across time
			 data_l / _r = left and right channels seperated
			 fs = sample rate
			 t = time vector corresponding to data
	""" 
	# filepath = '/Users/harrison/Desktop/Bass Line.wav'
	data, fs = librosa.load(filepath, sr=44100, mono=False)

	# Create a time vector for the audio
	t = np.linspace(0, (len(data)/fs), len(data))

	# Return all the goods
	return data, fs, t

def convert2db(x):
	return 20 * np.log10(x)

def normalize(x):
	x = x/(np.amax(np.absolute(x)))