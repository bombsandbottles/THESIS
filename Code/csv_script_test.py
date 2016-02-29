import csv
import feature_extraction
import numpy as np
np.set_printoptions(threshold='nan')
import librosa
import shutil
import mir_utils
import os, sys

filepath = '/Users/harrison/Desktop/Thesis_Test/Ariana Grande - One Last Time/converted_tracks_16_44100/normalized_tracks'

for stem in os.listdir(filepath):
	file_name, file_extension = os.path.splitext(stem)
	if file_extension == '.wav':
		# Get discretized audio to run through features
		print "Importing %s for analysis" % file_name
		data, fs, t = feature_extraction.import_audio(filepath + "/" + stem)

		# Calculate Dynamics Features
		print "Calculating dynamics features for %s" % file_name
		loudness_momentary, t_ = feature_extraction.calc_loudness(data, measurement = 'momentary')
		print "done!"
		loudness_short, t_ = feature_extraction.calc_loudness(data, measurement = 'short')
		print "done!"
		loudness_integrated = feature_extraction.calc_loudness(data, measurement = 'integrated')
		print "done!"
		lra = feature_extraction.calc_loudness(data, measurement = 'lra')
		print "done!"
		crest_factor_1s, t_ = feature_extraction.calc_crest_factor(data, win_size=1000, fs=44100)
		print "done!"
		crest_factor_100ms, t_ = feature_extraction.calc_crest_factor(data, win_size=100, fs=44100)
		print "done!"

		# Calculate Spectral Features
		print "Calculating spectral features for %s" % file_name
		spectral_centroid = feature_extraction.calc_spectral_centroid(data, win_size=2048)
		print "done!"
		spectral_spread = feature_extraction.calc_spectral_spread(data, win_size=2048)
		print "done!"
		spectral_skewness = feature_extraction.calc_spectral_skewness(data, win_size=2048)
		print "done!"
		spectral_kurtosis = feature_extraction.calc_spectral_kurtosis(data, win_size=2048)
		print "done!"
		zcr = feature_extraction.calc_zcr(data, win_size=2048, overlap=0, fs=44100)
		print "done!"
		spectral_rolloff = feature_extraction.calc_spectral_rolloff(data, win_size=2048, rolloff=0.85)
		print "done!"
		spectral_crest_factor = feature_extraction.calc_spectral_crest_factor(data, win_size=2048)
		print "done!"
		spectral_flatness = feature_extraction.calc_spectral_flatness(data, win_size=2048)
		print "done!"
		spectral_flux = feature_extraction.calc_spectral_flux(data, win_size=2048)
		print "done!"
		
		data = [ 
		['loudness_momentary', loudness_momentary],
		['loudness_short', loudness_short],
		['loudness_integrated', loudness_integrated],
		['lra', lra],
		['crest_factor_1s', crest_factor_1s],
		['crest_factor_100ms', crest_factor_100ms],
		['spectral_centroid', spectral_centroid],
		['spectral_spread', spectral_spread],
		['spectral_skewness', spectral_skewness],
		['spectral_kurtosis', spectral_kurtosis],
		['zcr', zcr],
		['spectral_rolloff', spectral_rolloff],
		['spectral_crest_factor', spectral_crest_factor],
		['spectral_flatness', spectral_flatness],
		['spectral_flux', spectral_flux]
		]

		# Write to CSV File
		with open(filepath + "/" + file_name + ".csv", "wb") as stem_data:
			w = csv.writer(stem_data)
			print "Writing %s to csv" % file_name
			w.writerows(data)
		stem_data.close()

print "DONE!!!"