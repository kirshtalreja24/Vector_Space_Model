import math
from Queries import Queries
from procecssor import InvertedIndex


class VectorProcessing:

    def __init__(self):
        self.__docVectors = []

    def CreateVectors(self):
        processor = InvertedIndex()
        processor.documentProcessing()

        queryObj = Queries(processor)

        words = queryObj.index
        all_docs = sorted(queryObj.all_docs)
        N = len(all_docs)

        for doc_id in all_docs:
            docVec = []
            for term in words:
                postings = words[term]
                df = len(postings)

                if doc_id in postings:
                    tf = len(postings[doc_id])
                else:
                    tf = 0

                idf = math.log10(N / df) if df != 0 else 0
                docVec.append(tf * idf)

            self.__docVectors.append(docVec)

    def writeToFile(self):
        with open("documentVectors.txt", "w") as f:
            for vector in self.__docVectors:
                f.write(str(vector) + '\n')