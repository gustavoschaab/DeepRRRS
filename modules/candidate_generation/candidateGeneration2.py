from datetime import datetime
import numpy
import pickle
import pandas as pd
import csv
import logging
import os
from .featureCombination import EnumFeatureCombination
from collections import defaultdict

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

class CandidateGeneration(object):

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
        '''

        Constructor of the Class.        
        '''
        self.startTime = datetime.now()
        self.outputDirectory = outputDirectory
        dt = datetime.now()
        self.log_directory = self.outputDirectory + "/logs/execution_" + str(dt.year) + str(dt.month) + str(dt.day) + "-" + str(dt.hour) + str(dt.minute)
        os.mkdir(self.log_directory)
        
        #self.configureLogging()
        self.logger = logging.getLogger('CandidateGeneration')
        self.logger.info("********************")
        self.logger.info("CANDIDATE GENERATION")
        self.logger.info("********************")
        self.logger.info(datasetName)
        self.logger.info("Begin Candidate Genearation")
        self.logger.info("userEmbeddingsFile: " + userEmbeddingsFile)
        self.logger.info("textualEmbeddingsFile: " + textualEmbeddingsFile)
        self.logger.info("imageEmbeddingsFile: " + imageEmbeddingsFile)
        self.logger.info("categoricalEmbeddingsFile: " + categoricalEmbeddingsFile)
        self.logger.info("coOccurenceEmbeddingsFile: " + cooccurenceEmbeddingsFile)
        self.logger.info("ratingsFile: " + ratingsFile)
        self.logger.info("Log: " + self.log_directory)

        self.textualEmbeddings = {}
        self.imageEmbeddings = {}
        self.categoricalEmbeddings = {}
        self.cooccurenceEmbeddings = {}        
        self.dict_users = {}

        self.userEmbeddingsFile = userEmbeddingsFile
        self.textualEmbeddingsFile = textualEmbeddingsFile
        self.imageEmbeddingsFile = imageEmbeddingsFile
        self.categoricalEmbeddingsFile = categoricalEmbeddingsFile
        self.cooccurenceEmbeddingsFile = cooccurenceEmbeddingsFile
        
        self.ratingsFile = ratingsFile
        self.lb = LabelBinarizer()  
        self.featureCombination = featureCombination      
        self.epochs = epochs        
        self.datasetName = datasetName
        self.kfold = kfold
        self.ktop = ktop
        self.threshold = threshold

    def configureLogging(self):
        '''
        Method responsible for configure the logging. The output will be save in the ./logs directory.
        ''' 
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=self.log_directory+'/CandidateGeneration.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
       

    def openTextualEmbeddingsFile(self):
        self.logger.info("Started openTextualEmbeddingsFile")
        with open(self.textualEmbeddingsFile, 'rb') as f:
            self.textualEmbeddings = pickle.load(f)
        self.logger.info("Done openTextualEmbeddingsFile")
    
    def openUserEmbeddings(self):
        self.logger.info("Started openUserEmbeddings")
        with open(self.userEmbeddingsFile, 'rb') as u:
            self.dict_users = pickle.load(u)
        self.logger.info("Done openUserEmbeddings")

    def openRatingsFile(self):
        self.logger.info("Started openRatingsFile")
        lst = []
        with open(self.ratingsFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lst = list(reader)
        
        self.logger.info("Creating list of dense vectors")
        ratings_list = []
        for row in lst:
            l = []
            user_dense_vector = self.dict_users[row[0]]
            l.append(row[0])
            l.append(row[1])
            l.append(row[2])
            l.append(user_dense_vector)
            ratings_list.append(l)

        self.logger.info("Creating Dataframe")
        df = pd.DataFrame(ratings_list)
        df.columns = ['UserId', 'ProductId', 'Rating', 'UserVector']
        df[['Rating']] = df[['Rating']].apply(pd.to_numeric)
        
        self.logger.info("Done openRatingsFile")
        return df

    def generateXY(self, dataset):
        self.logger.info("Started generateXY")
        UserId = []
        ItemId = []
        UserVector = []
        ItemVector = []
        Y = []
        for index, row in dataset.iterrows():
            UserVector.append(row['UserVector'])
            ItemVector.append(self.textualEmbeddings[row['ProductId']])
            Y.append(row['Rating'])
            UserId.append(row['UserId'])
            ItemId.append(row['ProductId'])
       
        self.logger.info("Done generateXY")
        return numpy.asarray(UserVector), numpy.asarray(ItemVector), numpy.asarray(Y), numpy.asarray(UserId), numpy.asarray(ItemId)

    def binarizeOutput(self, Y):  
        self.logger.info("Started binarizeOutput")
        outputBinarized = self.lb.fit_transform(Y)
        self.logger.info("Done binarizeOutput")
        return outputBinarized

    def defineModel(self):
        self.logger.info("Started defineModel")
        modelOptimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        userInput = Input(shape=[300], name="user")
        userInput2 = Dense(150, activation='relu')(userInput) 

        textualInput = Input(shape=[300], name="textual")
        textualInput2 = Dense(150, activation='relu')(textualInput) 

        input_vecs = Concatenate()([userInput2, textualInput2])        
        input_vecs = Dropout(0.2)(input_vecs)
        x = Dense(300, activation='relu')(input_vecs)
        x = Dropout(0.2)(x)
        x = Dense(150, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Dense(len(self.lb.classes_), activation="softmax")(x)
        model = Model(inputs=[userInput, textualInput], outputs=y)
        model.compile(loss='mse', optimizer=modelOptimizer, metrics=['mae', 'mse'])
        self.logger.info(modelOptimizer)
        #plot_model(model, to_file=self.log_directory + '/model_plot.png', show_shapes=True, show_layer_names=True)

        self.logger.info("Done defineModel")
        return model

    def runNeuralNetwork(self):
        self.openTextualEmbeddingsFile()
        self.openUserEmbeddings()
        ratingsDataset = self.openRatingsFile()
        
        X1, X2, Y, UserId, ItemId = self.generateXY(ratingsDataset) 
        self.logger.info("Dataset size: " + str(len(ratingsDataset)))       
        
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

            scores = model.evaluate([X1[test], X2[test]], Y_test, verbose=0)
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
            model.save(self.log_directory + "/model_" + str(idx+1) + ".h5")

        self.logger.info("Mean MSE validation: " + str(numpy.mean(mseScores)) + " Standard Deviation: " + str(numpy.std(mseScores)))
        self.logger.info("Mean MAE validation: " + str(numpy.mean(maeScores)) + " Standard Deviation: " + str(numpy.std(maeScores)))
        self.logger.info("Mean Precision: " + str(numpy.mean(precisionScores)) + " Standard Deviation: " + str(numpy.std(precisionScores)))
        self.logger.info("Mean Recall: " + str(numpy.mean(recallScores)) + " Standard Deviation: " + str(numpy.std(recallScores)))

        html_string = self.generateHTMLReport(len(ratingsDataset),
                                mseScores,
                                maeScores,
                                precisionScores,
                                recallScores)

        with open(self.log_directory + '/report.html','w') as f:
            f.write(html_string)
            f.close()
        

    def generateDataframe(self, UserVector, ItemVector, ActualRating):
        self.logger.info("Start Generate Dataframe")
        columns = ['UserVector', 'ItemVector', 'Actual']
        data = []
        for item in range(len(ActualRating)):
            userVector = UserVector[item]
            itemVector = ItemVector[item]
            actual = ActualRating[item]
            data.append([userVector, itemVector, actual])
        return pd.DataFrame(data = data, columns = columns)

    def generateDatasetWithPredictedValues(self, testData, model):
        self.logger.info("Start Generate Dataset with Predicted Values")
        predicted = []
        for _, row in testData.iterrows():
            y = model.predict([[row['UserVector']],[row['ItemVector']]])
            calculated_rating = numpy.argmax(y)+1
            predicted.append(calculated_rating)

        testData['Predicted'] = predicted

        return testData
    
    def computeRankingMetrics(self, predictions, k=3, threshold=4):
        self.logger.info("Start Compute Ranking Metrics")
        # First map the predictions to each user.
        user_pred_true = defaultdict(list)
        for _, row in predictions.iterrows():
            user_pred_true[row['UserId']].append((row['Predicted'], row['Actual']))

        precisions = {}
        recalls = {}

        for uid, user_ratings in user_pred_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls
    
    def generateHTMLReport(self, dataset_size, mse_scores, mae_scores, precision_scores, recall_scores):
        self.logger.info("Generate HTML Report...")
        html_string = '''
            <html>
                <head>
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                    <style>body{ margin:0 100; background:whitesmoke; }</style>
                </head>
                <body>        
                    <h3>Summary Report: </h3>
                    <table class="table table-striped">
                        <th align="left">Description</th><th>Value</th>
                        <tr>
                            <td align="left">Execution Starts</td>
                            <td>''' + str(self.startTime) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Execution Finish</td>
                            <td>''' + str(datetime.now()) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Feature Combination</td>
                            <td>''' + str(self.featureCombination.value) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Dataset name</td>
                            <td>''' + self.datasetName + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Dataset size</td>
                            <td>''' + str(dataset_size) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Number of Epochs</td>
                            <td>''' + str(self.epochs) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">K-Fold</td>
                            <td>''' + str(self.kfold) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Top-k</td>
                            <td>'''+ str(self.ktop) +'''</td>
                        </tr>                     
                    </table>
                    <h3>Training neural network</h3>
                    <h4>Model</h4>
                    <img src="model_plot.png">
                    <table class="table table-striped">
                        <th align="left">Description</th><th>Value</th>
                        <tr>
                            <td align="left">Average MSE</td>
                            <td>''' + str(numpy.mean(mse_scores)) + '''</td>
                        </tr>                    
                        <tr>
                            <td align="left">Average MAE</td>
                            <td>''' + str(numpy.mean(mae_scores)) + '''</td>
                        </tr>
                        <tr>
                            <td align="left">Average Precision</td>
                            <td>'''+ str(numpy.mean(precision_scores)) +'''</td>
                        </tr> 
                        <tr>
                            <td align="left">Average Recall</td>
                            <td>'''+ str(numpy.mean(recall_scores)) +'''</td>
                        </tr> 
                        <tr>
                            <td align="left">Std MAE</td>
                            <td>'''+ str(numpy.std(mae_scores)) +'''</td>
                        </tr>
                        <tr>
                            <td align="left">Std MSE</td>
                            <td>''' + str(numpy.std(mse_scores)) + '''</td>
                        </tr>
                    </table>
                    <p></p>

                    <table class="table table-striped">
                    <th>K-Fold</th>
                    <th>MSE</th>
                    <th>MAE</th>
                    <th>Precision</th>
                    <th>Recall</th>
                     '''
        
        for i in range(10):
            html_string = html_string + '''<tr> 
                <td>''' + str(i+1) + '''</td>
                <td>''' + str(mse_scores[i]) + '''</td>
                <td>''' + str(mae_scores[i]) + '''</td>
                <td>''' + str(precision_scores[i]) + '''</td>
                <td>''' + str(recall_scores[i]) + '''</td>  </tr>'''

        html_string = html_string + '''  </table> <p></p>
                    <table class="table table-striped">
                    <th>Training MSE</th>
                    <tr>
                        <td><img src="mse_1.png" height="360" width="480"></td>
                        <td><img src="mse_2.png" height="360" width="480"></td>
                        <td><img src="mse_3.png" height="360" width="480"></td>
                    </tr>
                    <tr>
                        <td><img src="mse_4.png" height="360" width="480"></td>
                        <td><img src="mse_5.png" height="360" width="480"></td>
                        <td><img src="mse_6.png" height="360" width="480"></td>
                    </tr>
                    <tr>
                        <td><img src="mse_7.png" height="360" width="480"></td>
                        <td><img src="mse_8.png" height="360" width="480"></td>
                        <td><img src="mse_9.png" height="360" width="480"></td>
                    </tr>
                    <tr>                    
                        <td><img src="mse_10.png" height="360" width="480"></td>
                    </tr>
                    </table>
                    <p></p>
                    
                    <table class="table table-striped">
                    <th>Training MAE</th>
                    <tr>
                        <td><img src="mae_1.png" height="360" width="480"></td>
                        <td><img src="mae_2.png" height="360" width="480"></td>
                        <td><img src="mae_3.png" height="360" width="480"></td>
                    </tr>
                    <tr>
                        <td><img src="mae_4.png" height="360" width="480"></td>
                        <td><img src="mae_5.png" height="360" width="480"></td>
                        <td><img src="mae_6.png" height="360" width="480"></td>
                    </tr>
                    <tr>
                        <td><img src="mae_7.png" height="360" width="480"></td>
                        <td><img src="mae_8.png" height="360" width="480"></td>
                        <td><img src="mae_9.png" height="360" width="480"></td>
                    </tr>
                    <tr>                    
                        <td><img src="mae_10.png" height="360" width="480"></td>
                    </tr>
                    </table>
                    <p></p>                

                </body>
            </html>'''

        return html_string

    def saveHistory(self, history, fold, dataset):
        self.logger.info('Saving history...')
        #with open(self.log_directory +  "/test_dataset_" + str(fold) + ".pkl", 'wb') as f:
        #    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        with open(self.log_directory + "/mse_" + str(fold) + ".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)            
            wr.writerow(history.history['mean_squared_error'])

        with open(self.log_directory + "/mae_" + str(fold) + ".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)            
            wr.writerow(history.history['mean_absolute_error'])

        with open(self.log_directory + "/val_mse_" + str(fold) + ".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)            
            wr.writerow(history.history['val_mean_squared_error'])

        with open(self.log_directory + "/val_mae_" + str(fold) + ".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)            
            wr.writerow(history.history['val_mean_absolute_error'])
