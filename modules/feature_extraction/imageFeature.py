import logging
import struct
import pickle
import csv
import numpy as np

class ImageFeature(object):

    def __init__(self, inputFile, outputFile, description, ratingFile, enableLogging):
        if enableLogging:
            self.configureLogging()
        self.logger = logging.getLogger('ImageFeature')
        self.logger.info("*************")
        self.logger.info("IMAGE FEATURE")
        self.logger.info("*************")
        self.logger.info(description)
        self.logger.info("Begin ImageFeaturen")
        self.inputFile = inputFile
        self.imageEmbeddings = {}
        self.outputFile = outputFile
        self.description = description
        self.ratingFile = ratingFile
        self.unique_items = []

    def generateEmbeddings(self):
        self.logger.info("Started readImageFeatures")
        self.getUniqueItems()
        f = open(self.inputFile, 'rb')
        count = 0
        while True:
            asin = f.read(10)
            if len(asin) == 0: 
                break
            feature = []

            for i in range(4096):
                float_position = f.read(4)
                value = struct.unpack('f', float_position)
                feature.append(value[0])
            
            if asin.decode("utf-8") in self.unique_items:  
                self.imageEmbeddings[asin.decode("utf-8")] = feature
                count = count + 1
            
                if ((count % 100) == 0):
                    self.logger.info("Converted: " + str(count))
            

        self.saveToFile()

        self.logger.info("**********************")
        self.logger.info("FINISHED IMAGE FEATURE")
        self.logger.info("**********************")
        self.logger.info(self.description)

    def saveToFile(self):
        self.logger.info("Started saveToFile")
        with open(self.outputFile, 'wb') as f:
          pickle.dump(self.imageEmbeddings, f, pickle.HIGHEST_PROTOCOL)
        self.logger.info("Finished saveToFile")

    def getUniqueItems(self):
        self.logger.info("Open input file")
        with open(self.ratingFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lst = list(reader)

        self.logger.info("Get items")
        items = [row[1] for row in lst]
        a = np.array(items)
        self.logger.info("Get unique items")
        self.unique_items = np.unique(a)
        self.logger.info("Number of unique items:" + str(len(self.unique_items)))

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='ImageFeature.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
