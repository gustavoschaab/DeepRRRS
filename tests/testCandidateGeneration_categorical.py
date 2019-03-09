import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration2
from modules.candidate_generation import candidateGeneration_categorical
from modules.candidate_generation import featureCombination
from modules.feature_extraction import generateUserEmbeddings
from modules.feature_extraction import categoricalFeature

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181115_home.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

cgc = candidateGeneration_categorical.CandidateGenerationCategorical("../data/AmazonClothings/userEmbeddings300.pkl", 
"", 
"",
"../data/AmazonClothings/categoricalEmbeddings300.pkl",
"",
"../data/AmazonClothings/ratings_5_core.csv", 
featureCombination.EnumFeatureCombination.CATEGORICAL,
"../data/AmazonClothings", 
7,
"Clothings")
cgc.runNeuralNetwork()