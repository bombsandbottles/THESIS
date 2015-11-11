import feature_extraction
import shutil
import mir_utils
import os, sys

def normalize_multitracks(filepath):
	"""
	filepath: string pointing to audio file on disk

	This function takes a directory of .wav multi-track files 
	and normalizes them such that the track with the highest peak-level
	is normalized as normal, and then all other tracks are raised
	the same amount of dB as the track with the highest peak-level.
	
	returns: A new directory of normalized data
	"""	
	peak_amplitudes = []
	tracks = os.listdir(filepath)
	norm_path = filepath + "/normalized_tracks"

	# Create the normalized data folder to house all our normalized data
	create_folder(norm_path)

	# Scan tracks, and get the peak amplitudes of each track
	for track in tracks:
		file_name, file_extension = os.path.splitext(track)
		if file_extension == '.wav':
			data_s, data_l, data_r, fs, t = feature_extraction.import_audio(filepath + "/" + track)
			peak_amplitude = calc_peak(data_l, data_r)
			peak_amplitudes.append(peak_amplitude)

	# Find the highest amplitude track by index
	idxs_norm = np.argwhere(peak_amplitudes == np.amax(peak_amplitudes))
	idxs_norm = idxs_norm.flatten().tolist()

	# Normalize the highest amplitude tracks and write them out
	track_num = -1
	for track in tracks:
		file_name, file_extension = os.path.splitext(track)
		if file_extension == '.wav': 
			track_num += 1
			if track_num in idxs:
				data_s, data_l, data_r, fs, t = feature_extraction.import_audio(filepath + "/" + track)
				write_path = (norm_path + "/" + file_name + "_normalized" + file_extension)
				librosa.output.write_wav(write_path, data_s, sr=44100, norm=True)

	# Find the indexes of all the other tracks that need to be raised relative to those normalized to 0
	idxs_relative = np.argwhere(peak_amplitudes != np.amax(peak_amplitudes))
	idxs_relative = idxs_relative.flatten().tolist()

	# This value will be used to scale all other tracks to those normalized
	scalar = np.amax(peak_amplitudes)

	# Raise the amplitude of all tracks the same amount that those normalized to 0 were raised
	track_num = -1
	for track in tracks:
		file_name, file_extension = os.path.splitext(track)
		if file_extension == '.wav':
			track_num += 1
			if track_num in idxs_relative:
				data_s, data_l, data_r, fs, t = feature_extraction.import_audio(filepath + "/" + track)
				# Relative Normalization
				data_s = data_s/scalar
				write_path = (norm_path + "/" + file_name + "_normalized" + file_extension)
				librosa.output.write_wav(write_path, data_s, sr=44100, norm=False)

def calc_peak(data_l, data_r):
	"""
	data_l, data_r: discrete audio arrays, left and right channels

	Quick function to find peak audio level of a stereo multi-track
	
	returns: the peak_amplitude 
	"""
	peaks = []
	peaks.append(np.amax(np.absolute(data_l)))
	peaks.append(np.amax(np.absolute(data_r)))
	if peaks[0] == peaks[1]:
		peak_amplitude = peaks[0]
	else:
		peak_amplitude = np.amax(peaks)
	return peak_amplitude

def create_folder(filepath):
	"""
	Creates the normalization folder to house all the new
	normalized data
	"""
	if os.path.exists(filepath):
		 shutil.rmtree(filepath)
	os.makedirs(filepath)

			