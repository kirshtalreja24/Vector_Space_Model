from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex
from Queries import Queries


processor = InvertedIndex()
processor.documentProcessing()
# processor.writeToFile()  
queryObj = Queries(processor)

queries = [
    "massive inflow of refugees",
    "pakistan afghanistan",
    "Hillary Clinton",
    "personnel policies",
    "united plane",
    "develop solutions",
    "developments praised",
    "muslims",
    "American Energy Revolution",
    "Future of new America",
    "Hillary clinton is the worst looser",
    "no patience for injustice",
    "Global interests",
    "pakistan afghanistan aid",
    "biggest plane wanted hour",
    "near architect box",
    "peaceful change",
]

for q in queries:
    queryObj.process_query(q)
    print()