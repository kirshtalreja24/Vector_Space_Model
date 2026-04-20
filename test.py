from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex
from Queries import Queries


processor = InvertedIndex()
processor.documentProcessing()
# processor.writeToFile()  
queryObj = Queries(processor)

