import batch_remove_sequence as brs
import batch_convert_data as bcd
import normalize_data as nd

def stem_data_munger(filepath, sequence=None, convert=True, normalize=True):
	
	if sequence != None:
		brs.batch_remove_sequence(filepath, sequence)

	if convert == True:
		converted_path = bcd.batch_convert(filepath)

	if  normalize == True:
		nd.normalize_multitracks(converted_path)

# -------------------------------------------------------------------------------- #
filepath = '/Volumes/LaCie/Remix Stems/Indabmusic:WAVO/ASTR Bleeding Love Stems'
sequence = 'ASTR_BLEEDINGLOVE_'

stem_data_munger(filepath, sequence, convert=True, normalize=True)