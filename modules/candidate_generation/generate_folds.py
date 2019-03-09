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

class GenerateFolds(CandidateGeneration):

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

        userInput = Input(shape=[4996], name="user")
        userInput2 = Dense(150, activation='relu')(userInput) 

        textualInput = Input(shape=[300], name="textual")
        textualInput2 = Dense(150, activation='relu')(textualInput) 

        categoricalInput = Input(shape=[300], name="categorical")
        categoricalInput2 = Dense(150, activation='relu')(categoricalInput) 

        coOccurenceInput = Input(shape=[300], name="coOccurence")
        coOccurenceInput2 = Dense(150, activation='relu')(coOccurenceInput) 

        imageInput = Input(shape=[4096], name="image")
        imageInput2 = Dense(150, activation='relu')(imageInput) 

        input_vecs = Concatenate()([userInput2, textualInput2, categoricalInput2, coOccurenceInput2, imageInput2])        
        input_vecs = Dropout(0.2)(input_vecs)
        x = Dense(750, activation='relu')(input_vecs)
        x = Dropout(0.2)(x)
        x = Dense(375, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Dense(len(self.lb.classes_), activation="softmax")(x)
        model = Model(inputs=[userInput, textualInput, categoricalInput, coOccurenceInput, imageInput], outputs=y)
        model.compile(loss='mse', optimizer=modelOptimizer, metrics=['mae', 'mse'])
        self.logger.info(modelOptimizer)
        #plot_model(model, to_file=self.log_directory + '/model_plot.png', show_shapes=True, show_layer_names=True)

        self.logger.info("Done defineModel")
        return model

    def openUserEmbeddings(self, paht):
        self.logger.info("Started openUserEmbeddings")
        with open(path, 'rb') as u:
            self.dict_users = pickle.load(u)
        self.logger.info("Done openUserEmbeddings")

    def generateFolds(self):
        self.openTextualEmbeddingsFile()
        self.openCategoricalEmbeddingsFile()
        self.openCooccurenceEmbeddingsFile()
        self.openUserEmbeddings()
        self.openImageEmbeddingsFile()
        ratingsDataset = self.openRatingsFile()
        
        X1, X2, X3, X4, X5, Y, UserId, ItemId = self.generateXY(ratingsDataset) 
        datasetSize = len(ratingsDataset)
        self.logger.info("Dataset size: " + str(datasetSize))       
        
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        
        for idx, (train, test) in enumerate(kfold.split(X1, Y)):            

            UserVectorTrain = X1[train]
            UserVectorTest = X1[test]
            TextualVectorTrain = X2[train]
            TextualVectorTest = X2[test]
            CategoricalVectorTrain = X3[train]
            CategoricalVectorTest = X3[test]
            CoOccurrenceVectorTrain = X4[train]
            CoOccurrenceVectorTest = X4[test]
            ImageVectorTrain = X5[train]
            ImageVectorTest = X5[test]

            Y_train = self.binarizeOutput(Y[train])
            Y_test = self.binarizeOutput(Y[test])

            UserIdTrain = UserId[train]
            UserIdTest = UserId[test]
            ItemIdTrain = ItemId[train]
            ItemIdTest = ItemId[test]

            with open('../data/AmazonElectronics/folds/UserVectorTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(UserVectorTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/UserVectorTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(UserVectorTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/TextualVectorTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(TextualVectorTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/TextualVectorTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(TextualVectorTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/CategoricalVectorTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(CategoricalVectorTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/CategoricalVectorTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(CategoricalVectorTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/CoOccurrenceVectorTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(CoOccurrenceVectorTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/CoOccurrenceVectorTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(CoOccurrenceVectorTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/ImageVectorTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(ImageVectorTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/ImageVectorTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(ImageVectorTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/YTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(Y_train, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/YTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(Y_test, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/UserIdTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(UserIdTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/UserIdTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(UserIdTest, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/ItemIdTrain'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(ItemIdTrain, f, pickle.HIGHEST_PROTOCOL)

            with open('../data/AmazonElectronics/folds/ItemIdTest'+str(idx)+'.pkl', 'wb') as f:
                pickle.dump(ItemIdTest, f, pickle.HIGHEST_PROTOCOL)


    def runNeuralNetwork(self):
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        mseScores = []
        maeScores = []
        precisionScores = []
        recallScores = []

        for i in range(self.kfold):
            print(i)

        for idx, (train, test) in enumerate(kfold.split(X1, Y)):
            Y_train = self.binarizeOutput(Y[train])
            Y_test = self.binarizeOutput(Y[test])
            model = self.defineModel()
            
            early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=2, restore_best_weights=True)

            self.logger.info("K-Fold: " + str(idx+1) + "/" + str(self.kfold))
            H = model.fit([X1[train], X2[train], X3[train], X4[train], X5[train]] , Y_train, validation_data=([X1[test], X2[test], X3[test], X4[test], X5[test]], Y_test), epochs=self.epochs, batch_size=32, callbacks=[early_stopping], verbose=2)
            
            scores = model.evaluate([X1[test], X2[test], X3[test], X4[test], X5[test]], Y_test, verbose=0)
            mseScores.append(scores[2])
            maeScores.append(scores[1])

            self.logger.info("Evaluation from Neural Network")
            self.logger.info("MSE: " + str(scores[2]))
            self.logger.info("MAE: " + str(scores[1]))
            self.logger.info(H.history.keys())

            testData = self.generateDataframe(UserId[test], ItemId[test], X1[test], X2[test], X3[test], X4[test], X5[test], Y[test])
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

    def openCategoricalEmbeddingsFile(self):
        self.logger.info("Started openCategoricalEmbeddingsFile")
        with open(self.categoricalEmbeddingsFile, 'rb') as f:
            self.categoricalEmbeddings = pickle.load(f)
        self.logger.info("Done openCategoricalEmbeddingsFile")

    def openCooccurenceEmbeddingsFile(self):
        self.logger.info("Started openCooccurenceEmbeddingsFile")
        with open(self.cooccurenceEmbeddingsFile, 'rb') as f:
            self.cooccurenceEmbeddings = pickle.load(f)
        self.logger.info("Done openCooccurenceEmbeddingsFile: " + str(len(self.cooccurenceEmbeddings)))

    def openImageEmbeddingsFile(self):
        self.logger.info("Started openImageEmbeddingsFile")
        with open(self.imageEmbeddingsFile, 'rb') as f:
            self.imageEmbeddings = pickle.load(f)
        self.logger.info("Done openImageEmbeddingsFile")

    def generateXY(self, dataset):
        self.logger.info("Started generateXY")
        UserId = []
        ItemId = []
        UserVector = []
        TextualVector = []
        CategoricalVector = []
        CoOccurenceVector = []
        ImageVector = []
        Y = []
        for _, row in dataset.iterrows():
            if row['ProductId'] in self.imageEmbeddings:
                UserVector.append(row['UserVector'])
                TextualVector.append(self.textualEmbeddings[row['ProductId']])
                CategoricalVector.append(self.categoricalEmbeddings[row['ProductId']])
                CoOccurenceVector.append(self.cooccurenceEmbeddings[row['ProductId']])
                ImageVector.append(self.imageEmbeddings[row['ProductId']])
                Y.append(row['Rating'])
                UserId.append(row['UserId'])
                ItemId.append(row['ProductId'])
       
        self.logger.info("Done generateXY")
        return numpy.asarray(UserVector), 
        numpy.asarray(TextualVector), 
        numpy.asarray(CategoricalVector), 
        numpy.asarray(CoOccurenceVector), 
        numpy.asarray(ImageVector),
        numpy.asarray(Y), 
        numpy.asarray(UserId), 
        numpy.asarray(ItemId)

    def generateDataframe(self, UserId, ItemId, UserVector, TextualVector, CategoricalVector, CoOccurenceVector, ImageVector, ActualRating):
        self.logger.info("Start Generate Dataframe")
        columns = ['UserId', 'ItemId', 'UserVector', 'TextualVector', 'CategoricalVector', 'CoOccurenceVector', 'ImageVector', 'Actual']
        data = []
        for item in range(len(UserId)):
            user = UserId[item]
            itemId = ItemId[item]
            userVector = UserVector[item]
            textualVector = TextualVector[item]
            categoricalVector = CategoricalVector[item]
            coOccurenceVector = CoOccurenceVector[item]
            imageVector = ImageVector[item]
            actual = ActualRating[item]
            data.append([user, itemId, userVector, textualVector, categoricalVector, coOccurenceVector, imageVector, actual])
        return pd.DataFrame(data = data, columns = columns)

    def generateDatasetWithPredictedValues(self, testData, model):
        self.logger.info("Start Generate Dataset with Predicted Values")
        predicted = []
        for _, row in testData.iterrows():
            y = model.predict([[row['UserVector']],[row['TextualVector']],[row['CategoricalVector']],[row['CoOccurenceVector']],[row['ImageVector']]])
            calculated_rating = numpy.argmax(y)+1
            predicted.append(calculated_rating)

        testData['Predicted'] = predicted

        return testData