import operator
import logging

class RankingGeneration(object):

    def __init__(self, predictedList, topN):
        self.configureLogging()
        self.logger = logging.getLogger('RankingGeneration')
        self.logger.info("Begin Ranking Genearation")

        self.predictedList = predictedList
        self.topN = topN

    def configureLogging(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='RankingGeneration.log',
                    filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def orderList(self):
        self.logger.info("orderList")
        return sorted(self.predictedList.items(), key=operator.itemgetter(1), reverse=True)

    def getTopN(self):
        self.logger.info("getTopN: " + str(self.topN))
        return self.orderList()[0:self.topN]