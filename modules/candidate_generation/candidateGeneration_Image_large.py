from .candidateGeneration2 import CandidateGeneration

import logging
import numpy
import pandas as pd
import pickle

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import Dot
from keras.layers import Input
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.models import load_model
#from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

class CandidateGenerationImageLarge(CandidateGeneration):

    def __init__(self, 
                 userEmbeddingsFile, 
                 textualEmbeddingsFile, 
                 imageEmbeddingsFile,
                 categoricalEmbeddingsFile,
                 cooccurenceEmbeddingsFile,
                 ratingsFile, 
                 featureCombination,
                 outputDirectory, 
                 epochs, 
                 datasetName, 
                 kfold=10, 
                 ktop=3, 
                 threshold=4):

        super().__init__(
                 userEmbeddingsFile, 
                 textualEmbeddingsFile, 
                 imageEmbeddingsFile,
                 categoricalEmbeddingsFile,
                 cooccurenceEmbeddingsFile,
                 ratingsFile, 
                 featureCombination,
                 outputDirectory, 
                 epochs, 
                 datasetName, 
                 kfold, 
                 ktop, 
                 threshold)
        
    def defineModel(self):
        self.logger.info("Started defineModel")
        modelOptimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        userInput = Input(shape=[4096], name="user")
        userInput2 = Dense(150, activation='relu')(userInput) 

        imageInput = Input(shape=[4096], name="image")
        imageInput2 = Dense(150, activation='relu')(imageInput) 

        input_vecs = Concatenate()([userInput2, imageInput2])        
        input_vecs = Dropout(0.2)(input_vecs)
        x = Dense(300, activation='relu')(input_vecs)
        x = Dropout(0.2)(x)
        x = Dense(150, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Dense(len(self.lb.classes_), activation="softmax")(x)
        model = Model(inputs=[userInput, imageInput], outputs=y)
        model.compile(loss='mse', optimizer=modelOptimizer, metrics=['mae', 'mse'])
        self.logger.info(modelOptimizer)
        #plot_model(model, to_file=self.log_directory + '/model_plot.png', show_shapes=True, show_layer_names=True)

        self.logger.info("Done defineModel")
        return model

    def testDatasetSize(self):
        self.openImageEmbeddingsFile()
        self.openUserEmbeddings()
        ratingsDataset = self.openRatingsFile()
        
        X1, X2, Y = self.generateXY(ratingsDataset)
        npX2 = numpy.asarray(X2)
        self.logger.info("Shape: " + str(npX2.shape))
        with open("np/X2.npy", 'wb') as f:
            numpy.save(f, npX2)
         

    def runNeuralNetwork(self):
        self.openImageEmbeddingsFile()
        self.openUserEmbeddings()
        ratingsDataset = self.openRatingsFile()
        datasetSize = len(ratingsDataset)

        X1, X2, Y = self.generateXY(ratingsDataset)        
        
        self.logger.info("Dataset size: " + str(datasetSize))
        del(ratingsDataset) 
        
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        mseScores = []
        maeScores = []
        precisionScores = []
        recallScores = []

        for idx, (train, test) in enumerate(kfold.split(X1, Y)):
            Y_train = self.binarizeOutput(Y[train])
            Y_test = self.binarizeOutput(Y[test])
            model = self.defineModel()            

            early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=2, restore_best_weights=True)

            self.logger.info("K-Fold: " + str(idx+1) + "/" + str(self.kfold))
            H = model.fit([X1[train], X2[train]] , Y_train, validation_data=([X1[test], X2[test]], Y_test), epochs=self.epochs, batch_size=32, callbacks=[early_stopping], verbose=2)
            
            scores = model.evaluate([X1[test], X2[test]], Y_test, verbose=2)
            mseScores.append(scores[2])
            maeScores.append(scores[1])

            self.logger.info("Evaluation from Neural Network")
            self.logger.info("MSE: " + str(scores[2]))
            self.logger.info("MAE: " + str(scores[1]))
            self.logger.info(H.history.keys())

            testData = self.generateDataframe(X1[test], X2[test], Y[test])
            testData = self.generateDatasetWithPredictedValues(testData, model)
            precisions, recalls = self.computeRankingMetrics(testData)

            precisionScores.append(sum([x for x in precisions.values()]) / sum([1 for x in precisions.values()]))
            recallScores.append(sum([x for x in recalls.values()]) / sum([1 for x in recalls.values()]))

            self.logger.info("Precision: " + str(sum([x for x in precisions.values()]) / sum([1 for x in precisions.values()])))
            self.logger.info("Recall: " + str(sum([x for x in recalls.values()]) / sum([1 for x in recalls.values()])))

            plt.plot(H.history['mean_squared_error'])
            plt.plot(H.history['val_mean_squared_error'])
            plt.title("K-Fold: " + str(idx+1))
            plt.ylabel("MSE")
            plt.xlabel("Epoch")
            plt.legend(["train", "test"], loc="upper right")
            plt.savefig(self.log_directory + "/mse_" + str(idx+1) + ".png")
            plt.cla()

            plt.plot(H.history['mean_absolute_error'])
            plt.plot(H.history['val_mean_absolute_error'])
            plt.title("K-Fold: " + str(idx+1))
            plt.ylabel("MAE")
            plt.xlabel("Epoch")
            plt.legend(["train", "test"], loc="upper right")
            plt.savefig(self.log_directory + "/mae_" + str(idx+1) + ".png")
            plt.cla()

            self.saveHistory(H, idx+1, testData)
            del(testData)
            model.save(self.log_directory + "/model_" + str(idx+1) + ".h5")

        self.logger.info("Mean MSE validation: " + str(numpy.mean(mseScores)) + " Standard Deviation: " + str(numpy.std(mseScores)))
        self.logger.info("Mean MAE validation: " + str(numpy.mean(maeScores)) + " Standard Deviation: " + str(numpy.std(maeScores)))
        self.logger.info("Mean Precision: " + str(numpy.mean(precisionScores)) + " Standard Deviation: " + str(numpy.std(precisionScores)))
        self.logger.info("Mean Recall: " + str(numpy.mean(recallScores)) + " Standard Deviation: " + str(numpy.std(recallScores)))

        html_string = self.generateHTMLReport(datasetSize,
                                mseScores,
                                maeScores,
                                precisionScores,
                                recallScores)

        with open(self.log_directory + '/report.html','w') as f:
            f.write(html_string)
            f.close()

    def openImageEmbeddingsFile(self):
        self.logger.info("Started openImageEmbeddingsFile")
        with open(self.imageEmbeddingsFile, 'rb') as f:
            self.imageEmbeddings = pickle.load(f)
        self.logger.info("Done openImageEmbeddingsFile")

    def generateXY(self, dataset):
        self.logger.info("Started generateXY")
        UserVector = []
        ItemVector = []
        Y = []
        for _, row in dataset.iterrows():
            if row['ProductId'] in self.imageEmbeddings:
                UserVector.append(row['UserVector'])
                ItemVector.append(self.imageEmbeddings[row['ProductId']])
                Y.append(row['Rating'])
                
        self.logger.info("Dataset size: " + str(len(Y)))
        # del(dataset)
        # npUserVector = numpy.asarray(UserVector)
        # del(UserVector)
        # npItemVector = numpy.asarray(ItemVector)
        # del(ItemVector)
        # npY = numpy.asarray(Y)
        # del(Y)
        
        self.logger.info("Done generateXY")

        return UserVector, ItemVector, Y
        #return npUserVector, npItemVector, npY