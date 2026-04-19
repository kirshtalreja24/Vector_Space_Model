import re
import unidecode
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import contractions
import DocumentExtraction


class InvertedIndex:
    def __init__(self):
        self.words = {}
        self.stopwords = set()
        self.stemmer = PorterStemmer()
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def readStopWords(self, filepath='Stopword-List.txt'):
        with open(filepath) as file:
            self.stopwords = set(line.strip().lower() for line in file)

    def clean_text(self, text):
        text = re.sub(' +', ' ', text)
        text = text.lower()
        text = unidecode.unidecode(text)
        return text

    def processWords(self, words):
        words = [contractions.fix(w) for w in words]

        expanded = []
        for w in words:
            expanded.extend(w.split())

        fixed = []
        for w in expanded:
            w = w.replace('-', ' ')
            fixed.extend(w.split())

        words = [w.translate(self.punctuation_table) for w in fixed]
        words = [w for w in words if w]

        words = [self.stemmer.stem(w) for w in words]
        words = [w for w in words if len(w) > 2 and w not in self.stopwords]

        return words

    def documentProcessing(self):
        fobj = DocumentExtraction.Extractedfiles()
        fobj.readData()
        files = fobj.getfiles()

        self.readStopWords()

        for doc_id, doc in enumerate(files, start=1):   # ✅ FIXED (start=1)
            text = self.clean_text(doc)

            tokens = word_tokenize(text)
            tokens = self.processWords(tokens)

            position = 0

            for word in tokens:
                if word not in self.words:
                    self.words[word] = {}

                if doc_id not in self.words[word]:
                    self.words[word][doc_id] = []

                self.words[word][doc_id].append(position)
                position += 1

        self.words = dict(sorted(self.words.items()))

    def processQuery(self, query):
        query = self.clean_text(query)
        tokens = word_tokenize(query)           # ✅ SAME as documents
        tokens = self.processWords(tokens)      # ✅ SAME pipeline
        return tokens