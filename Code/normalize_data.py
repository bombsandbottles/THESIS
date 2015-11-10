import feature_extraction
import os, sys

def normalize_multitracks(filepath):
	"""
	filepath: string pointing to audio file on disk

	This function takes a directory of .wav multi-track files 
	and normalizes them such that the track with the highest peak-level
	is normalized as normal, and then all other tracks are raised
	the same amount of dB as the track with the highest peak-level.
	
	returns: data = audio in samples across time
			 data_l / _r = left and right channels seperated
			 fs = sample rate
	 		 t = time vector corresponding to data
	"""	
	tracks = os.listdir(filepath)
        for track in tracks:	
        	file_name, file_extension = os.path.splitext(track)
        	if file_extension == '.wav':
        		data_s, data_l, data_r, fs, t = feature_extraction.import_audio(filepath + "/" + track)
        		"!!! TODO : grab peak audio level !!!"
        		

        		

            