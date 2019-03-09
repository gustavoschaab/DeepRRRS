import pickle
import logging
import numpy as np

class MergeEmbeddings(object):

    def __init__(self, textualEmbeddingsFile, categoricalEmbeddingsFile, imageEmbeddingsFile, coocurrenceEmbeddingsFile, outputEmbeddingsFile, description):
        self.configureLogging()
        self.logger = logging.getLogger('MergeEmbeddings')
        self.logger.info("****************")
        self.logger.info("MERGE EMBEDDINGS")
        self.logger.info("****************")
        self.logger.info(description)
        self.logger.info("Begin Merge Embeddings")
        self.logger.info("textualEmbeddingsFile: " + textualEmbeddingsFile)
        self.logger.info("categoricalEmbeddingsFile: " + categoricalEmbeddingsFile)
        self.logger.info("imageEmbeddingsFile: " + imageEmbeddingsFile)
        self.logger.info("coocurrenceEmbeddingsFile: " + coocurrenceEmbeddingsFile)

        self.textualEmbeddingsFile = textualEmbeddingsFile
        self.categoricalEmbeddingsFile = categoricalEmbeddingsFile
        self.imageEmbeddingsFile = imageEmbeddingsFile
        self.coocurrenceEmbeddingsFile = coocurrenceEmbeddingsFile
        self.outputEmbeddingsFile = outputEmbeddingsFile

        self.textualEmbeddings = {}
        self.categoricalEmbeddings = {}
        self.imageEmbeddings = {}
        self.coocurrenceEmbeddings = {}

        self.itemEmbeddings = {}

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/MergeEmbeddings.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def openFile(self, fileName):
        self.logger.info("Open File: " + fileName)
        if len(str(fileName))==0:
            return {}
        
        with open(fileName, 'rb') as u:
            return pickle.load(u)

    def merge(self):
        self.logger.info("Start merge")
        self.textualEmbeddings = self.openFile(self.textualEmbeddingsFile)
        self.imageEmbeddings = self.openFile(self.imageEmbeddingsFile)
        self.categoricalEmbeddings = self.openFile(self.categoricalEmbeddingsFile)
        self.coocurrenceEmbeddings = self.openFile(self.coocurrenceEmbeddingsFile)

        total = len(self.textualEmbeddings)
        self.logger.info("Total items: " + str(total))
        count = 0
        for key, value in self.categoricalEmbeddings.items():            
            embeddings = value
            #if (key in self.categoricalEmbeddings):
            #    embeddings = np.append(embeddings, self.categoricalEmbeddings[key])
            if (key in self.imageEmbeddings):
                embeddings = np.append(embeddings, self.imageEmbeddings[key])
            if (key in self.coocurrenceEmbeddings):
                embeddings = np.append(embeddings, self.coocurrenceEmbeddings[key])

            self.itemEmbeddings[key] = embeddings
            count = count + 1
            if (count%10000==0):
                self.logger.info("Progress.... " + str(count) + "/" + str(total))

        self.logger.info("Save file - name: " + self.outputEmbeddingsFile)
        with open(self.outputEmbeddingsFile, 'wb') as f:
            pickle.dump(self.itemEmbeddings, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("File saved")
        self.logger.info("Finished generateCategoricalEmbeddings")
