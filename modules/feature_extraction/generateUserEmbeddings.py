import csv
import logging
import pickle
import numpy as np

class User2Vec(object):

    def __init__(self, inputFile, outputFile, embeddingSize):
        #self.configureLogging()
        self.logger = logging.getLogger('User2Vec')
        self.logger.info("Begin User2Vec")
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.embeddingSize = embeddingSize
    
    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/User2Vec.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def generateEmbeddings(self):
        lst = []
        self.logger.info("Begin generateEmbeddings")
        self.logger.info("Open input file")
        with open(self.inputFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lst = list(reader)
        
        self.logger.info("Get users")
        users = [row[0] for row in lst]
        a = np.array(users)
        self.logger.info("Get unique users")
        unique_users = np.unique(a)

        self.logger.info("Total unique users" + str(len(unique_users)))
        user_dict = {}
        self.logger.info("Generate embeddings")
        count = 0
        for row in list(unique_users):
            user_dict[row] = np.random.uniform(low=-1.0, high=1.0, size=(self.embeddingSize,))
            count = count + 1
            if ((count%10000) == 0):
                self.logger.info("Generated: " + str(count))

        self.logger.info("Save file")
        with open(self.outputFile, 'wb') as f:
            pickle.dump(user_dict, f, pickle.HIGHEST_PROTOCOL)