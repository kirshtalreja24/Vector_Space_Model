from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex

files = Extractedfiles()
files.readData()
files = files.getfiles()
print(len(files))
print(files[0])


obj = InvertedIndex()
obj.documentProcessing()
# obj.writeToFile()



#get posting list for a specific word
arr = obj.getSpecificPostingList("hammer")
print(arr)
