from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import logging
import numpy as np
import ast
import csv
from nltk import RegexpTokenizer
import pickle

class CategoricalFeature(object):

    def __init__(self, productMetaFile, glove_input_file, outputEmbeddingsFile, ratingsFile="", enableLogging=False):
        if enableLogging:
            self.configureLogging()
        self.logger = logging.getLogger('CategoricalFeature')
        self.logger.info("Begin CategoricalFeature")
        self.productMetaFile = productMetaFile
        self.glove_input_file = glove_input_file
        self.outputEmbeddingsFile = outputEmbeddingsFile
        self.ratingsFile = ratingsFile
        self.categoricalEmbeddings = {}

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/CategoricalFeature.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def openRatingsFile(self):
        self.logger.info("Started openRatingsFile")
        lst = []
        with open(self.ratingsFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lst = list(reader)
        
        self.logger.info("Creating list of dense vectors")
        self.ratings_list = []
        for row in lst:
            if row[1] not in self.ratings_list:
                self.ratings_list.append(row[1])

        self.logger.info("Done openRatingsFile: " + str(len(self.ratings_list)))

    def prepareItemData(self):
        self.logger.info("Start prepareItemData")
        self.items = []

        self.logger.info("Read file: " + self.productMetaFile)
        count = 0
        with open (self.productMetaFile, 'rt') as in_file:
            for line in in_file:
                item = ast.literal_eval(line)
                if item['asin'] in self.ratings_list:                    
                    lst = []
                    lst.append(item['asin'])
                    lst.append(item['categories'])

                    self.items.append(lst)
                count = count + 1
                if ((count%10000)==0):
                    self.logger.info("Generated: "+str(count))
        
        self.logger.info("Finished prepareItemData: Items =>" + str(len(self.items)))

    def prepareWord2Vec(self):
        self.logger.info("Start prepareWord2Vec")
        word2vec_output_file = self.glove_input_file+'.word2vec'
        glove2word2vec(self.glove_input_file, word2vec_output_file)
        self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        self.logger.info("Finished prepareWord2Vec")
        
    def generateCategoricalEmbeddings(self):        
        self.logger.info("Start generateCategoricalEmbeddings")
        if self.ratingsFile != "":
            self.openRatingsFile()
        self.prepareItemData()
        self.prepareWord2Vec()

        count = 0
        tokenizer = RegexpTokenizer(r'\w+')
        for item in self.items:
            cat = []
            for categories in item[1]:        
                for word in categories:
                    new_word = word.lower()
                    d = tokenizer.tokenize(new_word)
                    cat = cat + d
                break
            embeddings = np.mean([self.model[x] for x in cat if x in self.model] or [np.zeros(300)], axis=0)
            self.categoricalEmbeddings[item[0]] = embeddings
            count = count + 1
            if ((count%10000)==0):
                self.logger.info("Generated: "+str(count))

        self.logger.info("Save file - Length: "+str(len(self.categoricalEmbeddings)))
        with open(self.outputEmbeddingsFile, 'wb') as f:
            pickle.dump(self.categoricalEmbeddings, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("File saved")
        self.logger.info("Finished generateCategoricalEmbeddings")
    