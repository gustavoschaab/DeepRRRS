import csv
import numpy as np
import logging

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from collections import defaultdict

class Baseline:
    pass

logger = logging.getLogger('Baselines')

def computeRankingMetrics(predictions, k=3, threshold=4):
    # First map the predictions to each user.
    user_pred_true = defaultdict(list)
    for row in predictions:
        user_pred_true[row[0]].append((float(row[3]), float(row[2])))

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

def openRatingsFile(path, delimiter):
    lst = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        lst = list(reader)
        
    return lst

def generateTestDataWithHash(calculatedData, originalData):
    logger.info("Generate Test Data")
    testData = []
    count = 0    
    print(len(calculatedData))
    print(len(originalData))
    for calculated in calculatedData:
        if calculated[0]+"_"+calculated[1] in originalData:
            count = count + 1
            testData.append(originalData[calculated[0]+"_"+calculated[1]] + [calculated[2]])
            if count%1000 == 0:
                logger.info("Generated: " + str(count))
    logger.info("Finish Test Data")
    return testData

def generateTestData(calculatedData, originalData):
    logger.info("Generate Test Data")
    testData = []
    count = 0
    for calculated in calculatedData:
        for original in originalData:
            if calculated[0] == original[0] and calculated[1] == original[1]:                
                count = count + 1
                testData.append(original + [calculated[2]])
                if count%1000 == 0:
                    logger.info("Generated: " + str(count))
                calculatedData.remove(calculated)
                break
    logger.info("Finish Test Data")
    return testData

def setupBaselines():
    logger.info("Setup baselines")
    baselines = []

    baseline01 = Baseline()
    baseline01.metric = "NMF"
    baseline01.dataset = "AmazonHome"
    baseline01.pathFullData = "../../data/AmazonHome/ratings_5_core.csv"
    baseline01.pathTestData = "D:\\git\\librec\\result\\AmazonHome-nmf-output\\nmf"
    baseline01.delimiterFullData = "\t"
    baseline01.delimiterTestData = ","
    baseline01.MSE = 0
    baseline01.MAE = 0
    baseline01.Precision = 0
    baseline01.Recall = 0
    baselines.append(baseline01)

    # baseline02 = Baseline()
    # baseline02.metric = "MostPopular"
    # baseline02.dataset = "AmazonHome"
    # baseline02.pathFullData = "../../data/AmazonHome/ratings_5_core.csv"
    # baseline02.pathTestData = "D:\\git\\librec\\result\\AmazonHome-mostpopular-output\\mostpopular"
    # baseline02.delimiterFullData = "\t"
    # baseline02.delimiterTestData = ","
    # baseline02.MSE = 0
    # baseline02.MAE = 0
    # baseline02.Precision = 0
    # baseline02.Recall = 0
    # baselines.append(baseline02)

    # baseline03 = Baseline()
    # baseline03.metric = "PMF"
    # baseline03.dataset = "AmazonHome"
    # baseline03.pathFullData = "../../data/AmazonHome/ratings_5_core.csv"
    # baseline03.pathTestData = "D:\\git\\librec\\result\\AmazonHome-pmf-output\\pmf"
    # baseline03.delimiterFullData = "\t"
    # baseline03.delimiterTestData = ","
    # baseline03.MSE = 0
    # baseline03.MAE = 0
    # baseline03.Precision = 0
    # baseline03.Recall = 0
    # baselines.append(baseline03)

    # baseline04 = Baseline()
    # baseline04.metric = "SVD++"
    # baseline04.dataset = "AmazonHome"
    # baseline04.pathFullData = "../../data/AmazonHome/ratings_5_core.csv"
    # baseline04.pathTestData = "D:\\git\\librec\\result\\AmazonHome-svdpp-output\\svdpp"
    # baseline04.delimiterFullData = "\t"
    # baseline04.delimiterTestData = ","
    # baseline04.MSE = 0
    # baseline04.MAE = 0
    # baseline04.Precision = 0
    # baseline04.Recall = 0
    # baselines.append(baseline04)

    # baseline05 = Baseline()
    # baseline05.metric = "NMF"
    # baseline05.dataset = "AmazonHealth"
    # baseline05.pathFullData = "../../data/AmazonHealth/ratings_5_core.csv"
    # baseline05.pathTestData = "D:\\git\\librec\\result\\AmazonHealth-nmf-output\\nmf"
    # baseline05.delimiterFullData = "\t"
    # baseline05.delimiterTestData = ","
    # baseline05.MSE = 0
    # baseline05.MAE = 0
    # baseline05.Precision = 0
    # baseline05.Recall = 0
    # baselines.append(baseline05)

    # baseline06 = Baseline()
    # baseline06.metric = "MostPopular"
    # baseline06.dataset = "AmazonHealth"
    # baseline06.pathFullData = "../../data/AmazonHealth/ratings_5_core.csv"
    # baseline06.pathTestData = "D:\\git\\librec\\result\\AmazonHealth-mostpopular-output\\mostpopular"
    # baseline06.delimiterFullData = "\t"
    # baseline06.delimiterTestData = ","
    # baseline06.MSE = 0
    # baseline06.MAE = 0
    # baseline06.Precision = 0
    # baseline06.Recall = 0
    # baselines.append(baseline06)

    # baseline07 = Baseline()
    # baseline07.metric = "PMF"
    # baseline07.dataset = "AmazonHealth"
    # baseline07.pathFullData = "../../data/AmazonHealth/ratings_5_core.csv"
    # baseline07.pathTestData = "D:\\git\\librec\\result\\AmazonHealth-pmf-output\\pmf"
    # baseline07.delimiterFullData = "\t"
    # baseline07.delimiterTestData = ","
    # baseline07.MSE = 0
    # baseline07.MAE = 0
    # baseline07.Precision = 0
    # baseline07.Recall = 0
    # baselines.append(baseline07)

    # baseline08 = Baseline()
    # baseline08.metric = "SVD++"
    # baseline08.dataset = "AmazonHealth"
    # baseline08.pathFullData = "../../data/AmazonHealth/ratings_5_core.csv"
    # baseline08.pathTestData = "D:\\git\\librec\\result\\AmazonHealth-svdpp-output\\svdpp"
    # baseline08.delimiterFullData = "\t"
    # baseline08.delimiterTestData = ","
    # baseline08.MSE = 0
    # baseline08.MAE = 0
    # baseline08.Precision = 0
    # baseline08.Recall = 0
    # baselines.append(baseline08)

    # baseline09 = Baseline()
    # baseline09.metric = "NMF"
    # baseline09.dataset = "AmazonElectronics"
    # baseline09.pathFullData = "../../data/AmazonElectronics/ratings_5_core.csv"
    # baseline09.pathTestData = "D:\\git\\librec\\result\\AmazonElectronics-nmf-output\\nmf"
    # baseline09.delimiterFullData = "\t"
    # baseline09.delimiterTestData = ","
    # baseline09.MSE = 0
    # baseline09.MAE = 0
    # baseline09.Precision = 0
    # baseline09.Recall = 0
    # baselines.append(baseline09)

    # baseline10 = Baseline()
    # baseline10.metric = "MostPopular"
    # baseline10.dataset = "AmazonElectronics"
    # baseline10.pathFullData = "../../data/AmazonElectronics/ratings_5_core.csv"
    # baseline10.pathTestData = "D:\\git\\librec\\result\\AmazonElectronics-mostpopular-output\\mostpopular"
    # baseline10.delimiterFullData = "\t"
    # baseline10.delimiterTestData = ","
    # baseline10.MSE = 0
    # baseline10.MAE = 0
    # baseline10.Precision = 0
    # baseline10.Recall = 0
    # baselines.append(baseline10)

    # baseline11 = Baseline()
    # baseline11.metric = "PMF"
    # baseline11.dataset = "AmazonElectronics"
    # baseline11.pathFullData = "../../data/AmazonElectronics/ratings_5_core.csv"
    # baseline11.pathTestData = "D:\\git\\librec\\result\\AmazonElectronics-pmf-output\\pmf"
    # baseline11.delimiterFullData = "\t"
    # baseline11.delimiterTestData = ","
    # baseline11.MSE = 0
    # baseline11.MAE = 0
    # baseline11.Precision = 0
    # baseline11.Recall = 0
    # baselines.append(baseline11)

    # baseline12 = Baseline()
    # baseline12.metric = "SVD++"
    # baseline12.dataset = "AmazonElectronics"
    # baseline12.pathFullData = "../../data/AmazonElectronics/ratings_5_core.csv"
    # baseline12.pathTestData = "D:\\git\\librec\\result\\AmazonElectronics-svdpp-output\\svdpp"
    # baseline12.delimiterFullData = "\t"
    # baseline12.delimiterTestData = ","
    # baseline12.MSE = 0
    # baseline12.MAE = 0
    # baseline12.Precision = 0
    # baseline12.Recall = 0
    # baselines.append(baseline12)

    logger.info("Finish Setup Baselines")

    return baselines

def generateDictionary(listRatings):
    d = {}
    for data in listRatings:
        d[data[0]+"_"+data[1]] = data
    return d

def main():    
    configureLogging()
    logger.info("Start")
    baselines = setupBaselines()
    for baseline in baselines:
        logger.info("Baseline: " + baseline.metric + " -- Dataset: " + baseline.dataset)
        originalData = openRatingsFile(baseline.pathFullData, baseline.delimiterFullData)
        calculatedData = openRatingsFile(baseline.pathTestData, baseline.delimiterTestData)
        hashtableOriginal = generateDictionary(originalData)
        logger.info("Size Original Data: " + str(len(originalData)))
        logger.info("Size Calculated Data: " + str(len(calculatedData)))

        testData = generateTestDataWithHash(calculatedData, hashtableOriginal)
        
        x = np.asarray(testData)

        y_true = x[:,2]
        y_pred = x[:,3]
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)

        baseline.MSE = mean_squared_error(y_true, y_pred)
        baseline.MAE = mean_absolute_error(y_true, y_pred)

        precisions, recalls = computeRankingMetrics(x)

        baseline.Precision = sum([x for x in precisions.values()]) / sum([1 for x in precisions.values()])
        baseline.Recall = sum([x for x in recalls.values()]) / sum([1 for x in recalls.values()])

        logger.info("Summary")
        logger.info("-------")
        logger.info("Baseline: " + baseline.metric)
        logger.info("Dataset: " + baseline.dataset)
        logger.info("MSE: " + str(baseline.MSE))
        logger.info("MAE: " + str(baseline.MAE))
        logger.info("Precision: " + str(baseline.Precision))
        logger.info("Recall: " + str(baseline.Recall))
        logger.info("************************************")
        logger.info("")

    
def configureLogging():
    '''
    Method responsible for configure the logging. The output will be save in the ./logs directory.
    ''' 
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename='baselines.log',
                filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == "__main__":
    main()