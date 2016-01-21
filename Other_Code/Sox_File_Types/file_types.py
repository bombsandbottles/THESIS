filepath = '/Users/harrison/Google Drive/NYU/Masters Music Tech/1 - THESIS/THESIS/Other_Code/Sox_File_Types/file_types.txt'
filepath_2 = '/Users/harrison/Google Drive/NYU/Masters Music Tech/1 - THESIS/THESIS/Other_Code/Sox_File_Types/file_types_edited.txt'

infile = open(filepath)
outfile = open(filepath_2, 'w')
outfile.truncate()

for word in infile.read().split():
	outfile.write('\'.' + word + '\',')
	outfile.write(' ')
outfile.close()