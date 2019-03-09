import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration2
from modules.candidate_generation import candidateGeneration_CategoricalCoOccurence
from modules.candidate_generation import featureCombination
from modules.feature_extraction import generateUserEmbeddings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181119_home.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# u2v = generateUserEmbeddings.User2Vec("../data/AmazonHome/ratings_5_core.csv", 
# "../data/AmazonHome/userEmbeddings600.pkl",
# 600)
# u2v.generateEmbeddings()

cgc = candidateGeneration_CategoricalCoOccurence.CandidateGenerationCategoricalCoOccurence("../data/AmazonClothings/userEmbeddings600.pkl", 
"", 
"",
"../data/AmazonClothings/categoricalEmbeddings300.pkl",
"../data/AmazonClothings/coOccurenceEmbeddings300.pkl",
"../data/AmazonClothings/ratings_5_core.csv", 
featureCombination.EnumFeatureCombination.CATEGORICAL_COOCCURENCE,
"../data/AmazonClothings", 
15,
"Clothings")
cgc.runNeuralNetwork()