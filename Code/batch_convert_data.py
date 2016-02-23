import shutil
import os, sys
import subprocess

def create_folder(filepath):
	"""
	Creates the normalization folder to house all the new
	normalized data
	"""
	if os.path.exists(filepath):
		 shutil.rmtree(filepath)
	os.makedirs(filepath)

def batch_convert(filepath, out_format='.wav', bit_depth=16, fs=44100):

	# Initialization
	tracks = os.listdir(filepath)
	if out_format == '.mp3':
		converted_path = filepath + "/converted_tracks" + "_320kbps"
	else:
		converted_path = filepath + "/converted_tracks" + "_" + str(bit_depth) + "_" + str(fs)

	# SOX file types
	file_types = ['.8svx', '.aif', '.aifc', '.aiff', '.aiffc', '.al', 
	'.amb', '.amr-nb', '.amr-wb', '.anb', '.au', '.avr', '.awb', '.caf', 
	'.cdda', '.cdr', '.cvs', '.cvsd', '.cvu', '.dat', '.dvms', '.f32', 
	'.f4', '.f64', '.f8', '.fap', '.flac', '.fssd', '.gsm', '.gsrt', '.hcom', 
	'.htk', '.ima', '.ircam', '.la', '.lpc', '.lpc10', '.lu', '.mat', '.mat4', 
	'.mat5', '.maud', '.mp2', '.mp3', '.nist', '.ogg', '.paf', '.prc', '.pvf', 
	'.raw', '.s1', '.s16', '.s2', '.s24', '.s3', '.s32', '.s4', '.s8', '.sb', 
	'.sd2', '.sds', '.sf', '.sl', '.sln', '.smp', '.snd', '.sndfile', '.sndr', 
	'.sndt', '.sou', '.sox', '.sph', '.sw', '.txw', '.u1', '.u16', '.u2', '.u24', 
	'.u3', '.u32', '.u4', '.u8', '.ub', '.ul', '.uw', '.vms', '.voc', '.vorbis', 
	'.vox', '.w64', '.wav', '.wavpcm', '.wve', '.xa', '.xi']

	# mkdir /converted_tracks
	print("Creating folder to store converted data...")
	create_folder(converted_path)

	for track in tracks:
		file_name, file_extension = os.path.splitext(track)
		if file_extension in file_types:
			infile = filepath + "/" + track
			outfile = converted_path + "/" + file_name + "_converted" + out_format
			if out_format == '.mp3':
				sox_command = ['sox', '-G', infile, '-C', '320', outfile, '-S']
			else:
				sox_command = ['sox', '-G', infile, '-b', str(bit_depth), '-r', str(fs), outfile, '-S']
			print("Converting %s with SOX...") % track
			subprocess.call(sox_command)