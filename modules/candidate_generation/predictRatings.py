import logging
import pickle
import csv
import numpy
import operator
from keras.models import load_model

class PredictRatings(object):

    def __init__(self, userId, modelFile, userEmbeddingsFile, textualEmbeddingsFile, ratingsFile):
        self.configureLogging()
        self.logger = logging.getLogger('PredictRatings')
        self.logger.info("Begin Predict Ratings")

        self.model = load_model(modelFile)
        self.textualEmbeddings = self.openTextualEmbeddingsFile(textualEmbeddingsFile)
        self.userEmbeddings = self.getUserEmbeddings(userId, userEmbeddingsFile)
        self.userItemsRatings = self.getUserItemsRatings(ratingsFile, userId)
        self.userId = userId

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='predictRatings.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def openTextualEmbeddingsFile(self, textualEmbeddingsFile):
        self.logger.info("Started openTextualEmbeddingsFile")
        with open(textualEmbeddingsFile, 'rb') as f:
            textualEmbeddings = pickle.load(f)
        self.logger.info("Done openTextualEmbeddingsFile")
        return textualEmbeddings

    def getUserEmbeddings(self, userId, userEmbeddingsFile):
        self.logger.info("Started getUserEmbeddings")
        
        with open(userEmbeddingsFile, 'rb') as u:
            dict_users = pickle.load(u)

        self.logger.info("UserId: " + str(userId))
        user = dict_users[userId]
        self.logger.info("User Embeddings: " + str(user))
        self.logger.info("Done getUserEmbeddings")
        return user

    def getUserItemsRatings(self, ratingsFile, userId):
        self.logger.info("Started getUserItemsRatings")
        
        lst = []
        with open(ratingsFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lst = list(reader)

        self.logger.info("Done getUserItemsRatings")
        return [item[1] for item in lst if item[0] == userId]
    
    def runPredictions(self):
        self.logger.info("Started runPredictions")

        self.logger.info("Removing rated items")
        for item in self.userItemsRatings:
            self.textualEmbeddings.pop(item, None)

        self.logger.info("Calculating predictions")
        self.logger.info("User: " + self.userId)
        predictions = {}        
        count = 0
        for key, value in self.textualEmbeddings.items():
            X = []
            embeddings = numpy.append(value, self.userEmbeddings) 
            X.append(embeddings)
            X = numpy.asarray(X)
            pred = self.model.predict(X)
            predictions[key] = pred
            count = count + 1
            if ((count%1000) == 0):
                self.logger.info("Calculated: " + str(count))
        
        
        self.logger.info("Done runPredictions")
        return predictions

    def convertPredictions(self, predictions):
        self.logger.info("Started convertPredictions")
        pred = {}
        count = 0
        for key, value in predictions.items():            
            calculated_rating = (numpy.argmax(value) * numpy.max(value))+1
            pred[key] = calculated_rating
            count = count + 1
            if ((count%10000) == 0):
                self.logger.info("Calculated: " + str(count))

        self.logger.info("Done convertPredictions")
        return pred

