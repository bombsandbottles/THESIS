import imp
from sys import argv

converter = imp.load_source('batch_convert_data', '/Users/harrison/Google Drive/NYU/Masters Music Tech/1 - THESIS/THESIS/Code/batch_convert_data.py')

print "Filepath"

converter.batch_convert('/Users/harrison/Desktop/Thesis_Test/Kesha - Cmon')