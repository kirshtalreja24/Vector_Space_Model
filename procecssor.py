import re
import unidecode
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import DocumentExtraction
import nltk


class InvertedIndex:
    def __init__(self):
        self.words = {}
        self.stopwords = set()
        self.lemmatizer = WordNetLemmatizer()
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
        # Expand contractions (e.g., don't -> do not)
        words = [contractions.fix(w) for w in words]

        # Split expanded contractions
        expanded = []
        for w in words:
            expanded.extend(w.split())

        # Handle hyphenated words
        fixed = []
        for w in expanded:
            w = w.replace('-', ' ')
            fixed.extend(w.split())

        words = [w.translate(self.punctuation_table) for w in fixed]
        words = [w for w in words if w]

        words = [self.lemmatizer.lemmatize(w) for w in words]

        words = [w for w in words if len(w) > 2 and w not in self.stopwords]

        return words

    def documentProcessing(self):
        fobj = DocumentExtraction.Extractedfiles()
        fobj.readData()
        files = fobj.getfiles()

        self.readStopWords()

        for doc_id, doc in enumerate(files, start=0):
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

        # Sort dictionary
        self.words = dict(sorted(self.words.items()))

    def writeToFile(self, filename="inverted_index.txt"):
        with open(filename, "w") as f:
            for word, postings in sorted(self.words.items()):
                f.write(f"{word} -> {postings}\n")

    def processQuery(self, query):
        query = self.clean_text(query)
        tokens = word_tokenize(query)
        tokens = self.processWords(tokens)
        return tokens

