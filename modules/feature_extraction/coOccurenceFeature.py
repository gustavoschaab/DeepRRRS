import logging
import numpy as np
import ast
from gensim.models import Word2Vec
import pickle
import csv

class CoOccurenceFeature(object):
    
    def __init__(self, productMetaFile, outputEmbeddingsFile, modelFile, ratingsFile="", enableLogging=False):
        if enableLogging:
            self.configureLogging()
        self.logger = logging.getLogger('CoOccurenceFeature')
        self.logger.info("Begin CoOccurenceFeature")
        self.productMetaFile = productMetaFile
        self.outputEmbeddingsFile = outputEmbeddingsFile
        self.modelFile = modelFile
        self.ratingsFile = ratingsFile
        self.cooccurenceEmbeddings = {}

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/CoOccurenceFeature.log',
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
                    if 'related' in item:
                        lst.append(item['related'])

                    self.items.append(lst)
                
                count = count + 1
                if ((count%10000)==0):
                    self.logger.info("Generated: "+str(count))
                    
        self.logger.info("Finised prepareItemData: Items =>" + str(len(self.items)))

    def trainWord2Vec(self):
        self.logger.info("Start trainWord2Vec: Items =>" + str(len(self.items)))
        sentences = []
        for item in self.items:
            if len(item) > 1:
                d = item[1]
                if 'also_viewed' in d:
                    sentences.append(d['also_viewed'])
                if 'also_bought' in d:
                    sentences.append(d['also_bought'])
                if 'bought_together' in d:
                    sentences.append(d['bought_together'])
        
        self.model = Word2Vec(sentences, min_count=1,size=300)
        self.logger.info("Model: ")
        self.logger.info(self.model)
        self.logger.info("Number of words: " + str(len(list(self.model.wv.vocab))))
        self.logger.info("Saving model to: " + str(self.modelFile))
        self.model.save(self.modelFile)

    def generateCoOccurenceEmbeddings(self):
        self.logger.info("Start generateCoOccurenceEmbeddings")
        self.openRatingsFile()
        self.prepareItemData()
        self.trainWord2Vec()        
        count = 0

        for item in self.items:
            relatedItems = []
            if len(item) > 1:
                d = item[1]
                if 'also_viewed' in d:
                    relatedItems = d['also_viewed'].copy()
                if 'also_bought' in d:
                    relatedItems = d['also_bought'].copy()
                if 'bought_together' in d:
                    relatedItems = d['bought_together'].copy()

                
                embeddings = np.mean([self.model[x] for x in relatedItems if x in self.model] or [np.zeros(300)], axis=0)
                
            else:
                embeddings = np.zeros(300)
            
            self.cooccurenceEmbeddings[item[0]] = embeddings
            count = count + 1
            if ((count%10000)==0):
                self.logger.info("Generated: "+str(count))
        
        self.logger.info("Save file - Length: "+str(len(self.cooccurenceEmbeddings)))
        with open(self.outputEmbeddingsFile, 'wb') as f:
            pickle.dump(self.cooccurenceEmbeddings, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("File saved")
        self.logger.info("Finished generateCoOccurenceEmbeddings")