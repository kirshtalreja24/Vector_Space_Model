from DocumentExtraction import Extractedfiles

files = Extractedfiles()
files.readData()
files = files.getfiles()
print(len(files))
print(files[0])



