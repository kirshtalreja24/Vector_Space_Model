from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex
from Queries import Queries



files = Extractedfiles()
processor = InvertedIndex()
processor.documentProcessing()
queryObj = Queries(processor)

query = "personnel policies"

queryObj.process_query(query)
