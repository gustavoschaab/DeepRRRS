from datetime import datetime
import json
import numpy as np
import ast
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import logging

class TextualFeature(object):
    
    def __init__(self, inputFile, outputFile, modelFile, max_epochs, vector_size, alpha, enableLogging):
        if enableLogging:
            self.configureLogging()
        self.logger = logging.getLogger('TextualFeature')
        self.logger.info("Begin Textual Feature")
        self.inputFile = inputFile        
        self.outputFile = outputFile
        self.modelFile = modelFile
        self.max_epochs = max_epochs
        self.vector_size = vector_size
        self.alpha = alpha
        self.items = {}
    
    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/TextualFeature.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def prepareData(self):
        self.logger.info("Started prepareData")
        with open (self.inputFile, 'rt') as in_file:
            for line in in_file:
                item = ast.literal_eval(line)
                descr = ""
                if 'title' in item:
                    descr = item['title']
                if 'description' in item:
                    descr += " " + item['description']
                self.items[item['asin']] = descr
        self.logger.info("Finished prepareData")

    def tokenizerData(self):
        self.logger.info("Started tokenizerData")
        tokenizer = RegexpTokenizer(r'\w+')
        for key, value in self.items.items():
            new_str = value.lower()
            dlist = tokenizer.tokenize(new_str)
            self.items[key] = dlist
        self.logger.info("Finished tokenizerData")
    
    def taggedDoc(self):
        self.logger.info("Started taggedDoc")
        items_tagged = [TaggedDocument(words=value, tags=[str(key)]) for key, value in self.items.items()]
        self.logger.info("Finished taggedDoc")
        return items_tagged

    def trainModel(self, model, items_tagged):
        self.logger.info("Started Training")
        for epoch in range(self.max_epochs):
            self.logger.info("iteration {0}".format(epoch))
            model.train(items_tagged,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        self.logger.info("Finished Training")

        model.save(self.modelFile)

    def generateEmbeddings(self):
        self.logger.info("Started generateEmbeddings")
        self.prepareData()
        self.tokenizerData()

        items_tagged = self.taggedDoc()

        model = Doc2Vec(vector_size=self.vector_size,
                        alpha=self.alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)
        
        self.logger.info("Build Model started")
        model.build_vocab(items_tagged)
        self.logger.info("Build Model finised")
        
        self.trainModel(model, items_tagged)

        self.logger.info("Collecting results started")
        items_embeddings = {}
        for key in self.items:
            items_embeddings[key] = model.docvecs[key]
        self.logger.info("Collecting results finished")

        self.logger.info("Save embeddings started")
        with open(self.outputFile, 'wb') as f:
            pickle.dump(items_embeddings, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("Save embeddings finished")

        self.logger.info("Finished generateEmbeddings")

        return items_embeddings
