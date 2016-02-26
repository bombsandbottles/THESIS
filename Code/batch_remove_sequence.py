import os

def batch_remove_sequence(filepath, sequence):
	'''
	This function can be used to remove prefix or suffixes from a large batch of stems
	as is common.

	ex. Exporting all tracks from Ableton Live
	'''
	for file in os.listdir(filepath):
		filename, ext = os.path.splitext(file)
		if sequence in filename:
			filename = filename.replace(sequence, "")
		new_filename = filename + ext
		os.rename(os.path.join(filepath, file), os.path.join(filepath, new_filename))

# Run Me for Victory
filepath = '/Users/harrison/Music/Reason Projects/Bombs and Bottles Productions/2016 - Audiobrain/HAL16/HAL16_stems'
sequence = 'HAL_A_BrandTheme_HZ_r1'
batch_remove_sequence(filepath, sequence)