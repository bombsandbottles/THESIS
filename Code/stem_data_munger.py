import batch_remove_sequence as brs
import batch_convert_data as bcd
import normalize_data as nd

def stem_data_munger(filepath, sequence=None, convert=True, normalize=True):
	"""
	This function prepares your data to be analyzed.
	Uses batch_remove_sequence, batch_convert_data, and normalize_data

	Parameters
	----------
	filepath: The path to your data
	sequence: A string that might be present in all the data (common with multi-tracks)
	convert: The data has to be either 16 or 32 bit wave files, this will take care of that
	normalize: This will perform the relative normalization so that only one stem peaks at 0dB 
	"""
	
	if sequence != None:
		brs.batch_remove_sequence(filepath, sequence)

	if convert == True:
		converted_path = bcd.batch_convert(filepath)

	if  normalize == True and convert == True:
		nd.normalize_multitracks(converted_path)
	elif normalize == True:
		nd.normalize_multitracks(filepath)

# -------------------------------------------------------------------------------- #
filepath = '/Volumes/LaCie/Remix Stems/to_be_organized/The Rubens - Hoops'

stem_data_munger(filepath, sequence='Hoops_EJ Rmx Stems_', convert=True, normalize=True)