import csv
import feature_extraction
import numpy as np
import librosa
import shutil
import mir_utils
import os, sys

filepath = '/Users/harrison/Desktop/Thesis_Test/Kesha - Cmon/converted_tracks_16_44100/normalized_tracks'
tracks = os.listdir(filepath)

# TODO
for stem in tracks:
	file_name, file_extension = os.path.splitext(stem)

	if file_extension == '.wav':
		# Get discretized audio to run through features
		print "Importing %s for analysis" % file_name
		data, fs, t = feature_extraction.import_audio(filepath + "/" + stem)

		# Get momentary loudness
		print "Calculating loudness of %s" % file_name
		loudness_momentary, lk_t = feature_extraction.calc_loudness(data, measurement = 'momentary')
		
		# Write to CSV File
		with open(filepath + "/" + file_name + ".csv", "wb") as stem_data:
			w = csv.writer(stem_data)
			print "Writing %s to csv" % file_name
			w.writerow(loudness_momentary)
		stem_data.close()